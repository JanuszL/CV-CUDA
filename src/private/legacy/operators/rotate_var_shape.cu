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


template<typename T>
__global__ void rotate_linear(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst,
                              const double *d_aCoeffs_)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if(dst_x >= dst.cols[batch_idx] || dst_y >= dst.rows[batch_idx])
        return;
    int height = src.rows[batch_idx], width = src.cols[batch_idx];

    const double *d_aCoeffs = (const double *)((char *)d_aCoeffs_ + (sizeof(double) * 6) * batch_idx);
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
__global__ void rotate_nearest(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst,
                               const double *d_aCoeffs_)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if(dst_x >= dst.cols[batch_idx] || dst_y >= dst.rows[batch_idx])
        return;
    int height = src.rows[batch_idx], width = src.cols[batch_idx];

    const double *d_aCoeffs = (const double *)((char *)d_aCoeffs_ + (sizeof(double) * 6) * batch_idx);
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
                cuda_op::CubicFilter< cuda_op::BorderReader< cuda_op::Ptr2dVarShapeNHWC<T>, cuda_op::BrdReplicate<T> > >filteredSrc,
                cuda_op::Ptr2dVarShapeNHWC<T> dst, const double *d_aCoeffs_)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if(dst_x >= dst.cols[batch_idx] || dst_y >= dst.rows[batch_idx])
        return;
    int height = filteredSrc.src.ptr.rows[batch_idx], width = filteredSrc.src.ptr.cols[batch_idx];

    const double *d_aCoeffs = (const double *)((char *)d_aCoeffs_ + (sizeof(double) * 6) * batch_idx);
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
void rotate(const void **d_in, void **d_out, const double *d_aCoeffs,
            const int batch_size, const int *height, const int *width, const int *out_height, const int *out_width,
            const int max_out_height, const int max_out_width,
            const int interpolation, cudaStream_t stream, size_t *pitch_in,
            size_t *pitch_out)
{
    const int channels = VecTraits<T>::cn;
    dim3 blockSize(BLOCK, BLOCK/4, 1);
    dim3 gridSize(divUp(max_out_width, blockSize.x), divUp(max_out_height, blockSize.y), batch_size);
    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(batch_size, height, width, channels, (T **) d_in);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(batch_size, out_height, out_width, channels, (T **) d_out);
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
    else // cv::INTER_CUBIC
    {
        cuda_op::BrdReplicate<T> brd(0, 0);
        cuda_op::BorderReader< cuda_op::Ptr2dVarShapeNHWC<T>, cuda_op::BrdReplicate<T> > brdSrc(src_ptr, brd);
        cuda_op::CubicFilter< cuda_op::BorderReader< cuda_op::Ptr2dVarShapeNHWC<T>, cuda_op::BrdReplicate<T> > > filteredSrc(
                        brdSrc);

        rotate_cubic<T><<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr, d_aCoeffs);
        checkKernelErrors();
    }
}

namespace cuda_op
{

size_t RotateVarShape::calBufferSize(int batch_size)
{
    return (sizeof(void *) * 2 + sizeof(double) * 6 + sizeof(int) * 4 + sizeof(size_t) * 2) * batch_size;
}

int RotateVarShape::infer(const void **data_in, void **data_out, void *gpu_workspace, void *cpu_workspace,
                          const int batch, const size_t buffer_size, const cv::Size *dsize, const double *angle, const double *xShift,
                          const double *yShift, const int interpolation, DataShape *input_shape, DataFormat format, DataType data_type,
                          cudaStream_t stream)
{
    if(!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = input_shape[0].C;

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

    const void **inputs = (const void **)cpu_workspace;
    void **outputs = (void **)((char *)inputs + sizeof(void *) * batch);
    int *rows = (int *)((char *)outputs + sizeof(void *) * batch);
    int *cols = (int *)((char *)rows + sizeof(int) * batch);
    int *out_rows = (int *)((char *)cols + sizeof(int) * batch);
    int *out_cols = (int *)((char *)out_rows + sizeof(int) * batch);
    size_t *pitch_in = (size_t *)((char *)out_cols + sizeof(int) * batch);
    size_t *pitch_out = (size_t *)((char *)pitch_in + sizeof(size_t) * batch);
    double *d_aCoeffs = (double *)((char *)pitch_out + sizeof(size_t) * batch);

    size_t data_size = DataSize(data_type);
    int max_out_width = 0, max_out_height = 0;

    for(int i = 0; i < batch; ++i)
    {
        inputs[i] = data_in[i];
        outputs[i] = data_out[i];
        double tmp_angle = angle[i];
        d_aCoeffs[i*6] = cos(tmp_angle * PI / 180);
        d_aCoeffs[i*6+1] = sin(tmp_angle * PI / 180);
        d_aCoeffs[i*6+2] = xShift[i];
        d_aCoeffs[i*6+3] = -d_aCoeffs[i*6+1];
        d_aCoeffs[i*6+4] = d_aCoeffs[i*6];
        d_aCoeffs[i*6+5] = yShift[i];
        rows[i] = input_shape[i].H;
        cols[i] = input_shape[i].W;
        out_rows[i] = dsize[i].height;
        out_cols[i] = dsize[i].width;
        pitch_in[i] = cols[i] * channels * data_size;
        pitch_out[i] = dsize[i].width * channels * data_size;
        if(max_out_width < dsize[i].width)
            max_out_width = dsize[i].width;
        if(max_out_height < dsize[i].height)
            max_out_height = dsize[i].height;
    }

    const void **inputs_gpu = (const void **)gpu_workspace;
    void **outputs_gpu = (void **)((char *)inputs_gpu + sizeof(void *) * batch);
    int *rows_gpu = (int *)((char *)outputs_gpu + sizeof(void *) * batch);
    int *cols_gpu = (int *)((char *)rows_gpu + sizeof(int) * batch);
    int *out_rows_gpu = (int *)((char *)cols_gpu + sizeof(int) * batch);
    int *out_cols_gpu = (int *)((char *)out_rows_gpu + sizeof(int) * batch);
    size_t *pitch_in_gpu = (size_t *)((char *)out_cols_gpu + sizeof(int) * batch);
    size_t *pitch_out_gpu = (size_t *)((char *)pitch_in_gpu + sizeof(size_t) * batch);
    double *d_aCoeffs_gpu = (double *)((char *)pitch_out_gpu + sizeof(size_t) * batch);

    checkCudaErrors(cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, buffer_size, cudaMemcpyHostToDevice,
                                    stream));

    typedef void (*func_t)(const void **d_in, void **d_out, const double* d_aCoeffs, const int batch_size,
                           const int *height, const int *width, const int *out_height, const int *out_width, const int max_out_height,
                           const int max_out_width, const int interpolation, cudaStream_t stream, size_t *pitch_in, size_t *pitch_out);

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

    func(inputs_gpu, outputs_gpu, d_aCoeffs_gpu, batch, rows_gpu, cols_gpu, out_rows_gpu, out_cols_gpu, max_out_height,
         max_out_width, interpolation,
         stream, pitch_in_gpu, pitch_out_gpu);
    CV_Assert(func != 0);
    return 0;
}

} // cuda_op