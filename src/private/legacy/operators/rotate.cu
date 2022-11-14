/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cv_cuda.h"
#include "cuda_utils.cuh"
#include "border.cuh"

#define BLOCK 32
#define PI 3.1415926535897932384626433832795
using namespace cv::cudev;

__global__ void compute_warpAffine(const double angle, const double xShift, const double yShift, double aCoeffs[6])
{
    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

template<typename T>
__global__ void rotate_linear(const cuda_op::Ptr2dNHWC<T> src, cuda_op::Ptr2dNHWC<T> dst, const double d_aCoeffs[6])
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(dst_x >= dst.cols || dst_y >= dst.rows)
        return;
    const int batch_idx = get_batch_idx();
    int height = src.rows, width = src.cols;

    const double dst_x_shift = dst_x - d_aCoeffs[2];
    const double dst_y_shift = dst_y - d_aCoeffs[5];
    float src_x = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float src_y = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if(src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        typedef typename MakeVec<float, VecTraits<T>::cn>::type work_type;
        work_type out = VecTraits<work_type>::all(0);

        const int x1 = __float2int_rz(src_x);
        const int y1 = __float2int_rz(src_y);
        const int x2 = x1 + 1;
        const int y2 = y1 + 1;
        const int x2_read = min(x2, width - 1);
        const int y2_read = min(y2, height - 1);

        T src_reg = *src.ptr(batch_idx, y1, x1);
        out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *src.ptr(batch_idx, y1, x2_read);
        out = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *src.ptr(batch_idx, y2_read, x1);
        out = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *src.ptr(batch_idx, y2_read, x2_read);
        out = out + src_reg * ((src_x - x1) * (src_y - y1));

        *dst.ptr(batch_idx, dst_y, dst_x) = saturate_cast<T>(out);
    }
}

template<typename T>
__global__ void rotate_nearest(const cuda_op::Ptr2dNHWC<T> src, cuda_op::Ptr2dNHWC<T> dst, const double d_aCoeffs[6])
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(dst_x >= dst.cols || dst_y >= dst.rows)
        return;
    const int batch_idx = get_batch_idx();
    int height = src.rows, width = src.cols;

    const double dst_x_shift = dst_x - d_aCoeffs[2];
    const double dst_y_shift = dst_y - d_aCoeffs[5];

    float src_x = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float src_y = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if(src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        const int x1 = min(__float2int_rz(src_x + 0.5), width - 1);
        const int y1 = min(__float2int_rz(src_y + 0.5), height - 1);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, y1, x1);
    }
}

template<typename T>
__global__ void rotate_cubic(
                cuda_op::CubicFilter< cuda_op::BorderReader< cuda_op::Ptr2dNHWC<T>, cuda_op::BrdReplicate<T> > > filteredSrc,
                cuda_op::Ptr2dNHWC<T> dst, const double d_aCoeffs[6])
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(dst_x >= dst.cols || dst_y >= dst.rows)
        return;
    const int batch_idx = get_batch_idx();
    int height = filteredSrc.src.ptr.rows, width = filteredSrc.src.ptr.cols;

    const double dst_x_shift = dst_x - d_aCoeffs[2];
    const double dst_y_shift = dst_y - d_aCoeffs[5];

    float src_x = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
    float src_y = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

    if(src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
    {
        *dst.ptr(batch_idx, dst_y, dst_x) = filteredSrc(batch_idx, src_y, src_x);
    }
}

template<typename T> // uchar3 float3 uchar1 float3
void rotate(const void *d_in, void *d_out, const double angle, const double xShift, const double yShift,
            const int batch_size, const int height, const int width, const int out_height, const int out_width,
            double d_aCoeffs[6] /*device pointer*/,  const int interpolation, cudaStream_t stream)
{
    compute_warpAffine<<<1, 1, 0, stream>>>(angle, xShift, yShift, d_aCoeffs);
    checkKernelErrors();
    int channels = VecTraits<T>::cn;
    dim3 blockSize(BLOCK, BLOCK/4, 1);
    dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);
    cuda_op::Ptr2dNHWC<T> src_ptr(batch_size, height, width, channels, (T *) d_in);
    cuda_op::Ptr2dNHWC<T> dst_ptr(batch_size, out_height, out_width, channels, (T *) d_out);

    if(interpolation == cv::INTER_LINEAR)
    {
        rotate_linear<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, d_aCoeffs);
        checkKernelErrors();
    }
    else if(interpolation == cv::INTER_NEAREST)
    {
        rotate_nearest<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, d_aCoeffs);
        checkKernelErrors();
    }
    else if(interpolation == cv::INTER_CUBIC)
    {
        cuda_op::BrdReplicate<T> brd(src_ptr.rows, src_ptr.cols);
        cuda_op::BorderReader< cuda_op::Ptr2dNHWC<T>, cuda_op::BrdReplicate<T> > brdSrc(src_ptr, brd);
        cuda_op::CubicFilter< cuda_op::BorderReader< cuda_op::Ptr2dNHWC<T>, cuda_op::BrdReplicate<T> > > filteredSrc(brdSrc);

        rotate_cubic<T><<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr, d_aCoeffs);
        checkKernelErrors();
    }

    #ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
    #endif
}

namespace cuda_op
{

size_t Rotate::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 6 * sizeof(double);
}

int Rotate::infer(const void *const *inputs, void **outputs, void *workspace, const cv::Size dsize, const double angle,
                  const double xShift, const double yShift, const int interpolation, DataShape input_shape, DataFormat format,
                  DataType data_type, cudaStream_t stream)
{
    if(!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int batch = input_shape.N;
    int channels = input_shape.C;
    int rows = input_shape.H;
    int cols = input_shape.W;
    size_t data_size = DataSize(data_type);

    if(channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if(!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if(!(interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_NEAREST || interpolation == cv::INTER_CUBIC))
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const void *d_in, void *d_out, const double angle, const double xShift, const double yShift,
                           const int batch_size, const int height, const int width, const int out_height, const int out_width,
                           double d_aCoeffs[6] /*device pointer*/,  const int interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {rotate<uchar>, 0 /*rotate<uchar2>*/, rotate<uchar3>, rotate<uchar4>     },
        {0 /*rotate<schar>*/, 0 /*rotate<char2>*/, 0 /*rotate<char3>*/, 0 /*rotate<char4>*/},
        {rotate<ushort>, 0 /*rotate<ushort2>*/, rotate<ushort3>, rotate<ushort4>    },
        {rotate<short>, 0 /*rotate<short2>*/, rotate<short3>, rotate<short4>     },
        {0 /*rotate<int>*/, 0 /*rotate<int2>*/, 0 /*rotate<int3>*/, 0 /*rotate<int4>*/ },
        {rotate<float>, 0 /*rotate<float2>*/, rotate<float3>, rotate<float4>     }
    };

    const func_t func = funcs[data_type][channels - 1];
    CV_Assert(func != 0);

    func(inputs[0], outputs[0], angle, xShift, yShift, batch, rows, cols, dsize.height, dsize.width,
         (double *)workspace, interpolation, stream);
    return 0;
}

} // cuda_op