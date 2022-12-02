/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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
#include "transform.cuh"

#define BLOCK 32
using namespace cv::cudev;

template <class Transform, class Filter, typename T> __global__ void warp(const Filter src,
        cuda_op::Ptr2dVarShapeNHWC<T> dst,
        const float *d_coeffs_)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int lid = get_lid();
    const float *d_coeffs = d_coeffs_ + batch_idx * 9;

    extern __shared__ float coeff[];
    if(lid < 9)
    {
        coeff[lid] = d_coeffs[lid];
    }
    __syncthreads();

    if(x < dst.cols[batch_idx] && y < dst.rows[batch_idx])
    {
        const float2 coord = Transform::calcCoord(coeff, x, y);
        *dst.ptr(batch_idx, y, x) = saturate_cast<T>(src(batch_idx, coord.y, coord.x));
    }
}

template <class Transform, template <typename> class Filter, template <typename> class B, typename T> struct
    WarpDispatcher
{
    static void call(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst, float *d_coeffs,
                     const int max_height, const int max_width,
                     const float *borderValue, cudaStream_t stream)
    {
        typedef typename MakeVec<float, VecTraits<T>::cn>::type work_type;

        dim3 block(BLOCK, BLOCK/4);
        dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), dst.batches);

        B<work_type> brd(0, 0, VecTraits<work_type>::make(borderValue));
        cuda_op::BorderReader< cuda_op::Ptr2dVarShapeNHWC<T>, B<work_type> > brdSrc(src, brd);
        Filter< cuda_op::BorderReader< cuda_op::Ptr2dVarShapeNHWC<T>, B<work_type> > > filter_src(brdSrc);
        size_t smem_size = 9 * sizeof(float);
        warp<Transform><<<grid, block, smem_size, stream>>>(filter_src, dst, d_coeffs);
        checkKernelErrors();
    }
};

template <class Transform, typename T>
void warp_caller(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst, float *d_coeffs,
                 const int max_height, const int max_width,
                 int interpolation, int borderMode, const float *borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst, float* d_coeffs,
                           const int max_height, const int max_width, const float* borderValue, cudaStream_t stream);

    static const func_t funcs[3][5] =
    {
        {
            WarpDispatcher<Transform, cuda_op::PointFilter, cuda_op::BrdConstant, T>::call,
            WarpDispatcher<Transform, cuda_op::PointFilter, cuda_op::BrdReplicate, T>::call,
            WarpDispatcher<Transform, cuda_op::PointFilter, cuda_op::BrdReflect, T>::call,
            WarpDispatcher<Transform, cuda_op::PointFilter, cuda_op::BrdWrap, T>::call,
            WarpDispatcher<Transform, cuda_op::PointFilter, cuda_op::BrdReflect101, T>::call
        },
        {
            WarpDispatcher<Transform, cuda_op::LinearFilter, cuda_op::BrdConstant, T>::call,
            WarpDispatcher<Transform, cuda_op::LinearFilter, cuda_op::BrdReplicate, T>::call,
            WarpDispatcher<Transform, cuda_op::LinearFilter, cuda_op::BrdReflect, T>::call,
            WarpDispatcher<Transform, cuda_op::LinearFilter, cuda_op::BrdWrap, T>::call,
            WarpDispatcher<Transform, cuda_op::LinearFilter, cuda_op::BrdReflect101, T>::call
        },
        {
            WarpDispatcher<Transform, cuda_op::CubicFilter, cuda_op::BrdConstant, T>::call,
            WarpDispatcher<Transform, cuda_op::CubicFilter, cuda_op::BrdReplicate, T>::call,
            WarpDispatcher<Transform, cuda_op::CubicFilter, cuda_op::BrdReflect, T>::call,
            WarpDispatcher<Transform, cuda_op::CubicFilter, cuda_op::BrdWrap, T>::call,
            WarpDispatcher<Transform, cuda_op::CubicFilter, cuda_op::BrdReflect101, T>::call
        }
    };

    funcs[interpolation][borderMode](src, dst, d_coeffs, max_height, max_width, borderValue, stream);
}
namespace cuda_op
{

template<typename T>
void warpPerspective(const void **input, void **output, float *d_coeffs, const int *height, const int *width,
                     const int *out_height, const int *out_width, const int max_height, const int max_width, const int batch,
                     const int interpolation, int borderMode, const float *borderValue, cudaStream_t stream)
{
    int channels = VecTraits<T>::cn;
    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(batch, height, width, channels, (T **) input);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(batch, out_height, out_width, channels, (T **) output);

    warp_caller<PerspectiveTransform, T>(src_ptr, dst_ptr, d_coeffs, max_height, max_width,
                                         interpolation, borderMode, borderValue, stream);
}

size_t WarpPerspectiveVarShape::calBufferSize(int batch_size)
{
    return (2 * sizeof(void *) + 4 * sizeof(int) + 9 * sizeof(float)) * batch_size;
}

int WarpPerspectiveVarShape::infer(const void **data_in, void **data_out, void *gpu_workspace, void *cpu_workspace,
                                   const int batch, const size_t buffer_size, const cv::Size *dsize, const float *trans_matrix,
                                   const int flags, const int borderMode, const cv::Scalar borderValue, const DataShape *input_shape,
                                   DataFormat format, DataType data_type, cudaStream_t stream)
{
    if(!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = input_shape[0].C;
    const int interpolation = flags & cv::INTER_MAX;

    if(channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if(!(data_type == kCV_8U || data_type == kCV_8S ||
            data_type == kCV_16U || data_type == kCV_16S ||
            data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    CV_Assert(interpolation == cv::INTER_NEAREST || interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC);
    CV_Assert(borderMode == cv::BORDER_REFLECT101 || borderMode == cv::BORDER_REPLICATE
              || borderMode == cv::BORDER_CONSTANT || borderMode == cv::BORDER_REFLECT || borderMode == cv::BORDER_WRAP);

    const void **inputs = (const void **)cpu_workspace;
    void **outputs = (void **)((char *)inputs + sizeof(void *) * batch);
    int *rows = (int *)((char *)outputs + sizeof(void *) * batch);
    int *cols = (int *)((char *)rows + sizeof(int) * batch);
    int *out_rows = (int *)((char *)cols + sizeof(int) * batch);
    int *out_cols = (int *)((char *)out_rows + sizeof(int) * batch);
    float *d_aCoeffs = (float *)((char *)out_cols + sizeof(int) * batch);

    size_t data_size = DataSize(data_type);
    int max_out_width = 0, max_out_height = 0;

    for(int i = 0; i < batch; ++i)
    {
        inputs[i] = data_in[i];
        outputs[i] = data_out[i];
        rows[i] = input_shape[i].H;
        cols[i] = input_shape[i].W;
        out_rows[i] = dsize[i].height;
        out_cols[i] = dsize[i].width;

        cv::Mat coeffsMat(3, 3, CV_32FC1, (void *)(d_aCoeffs + i * 9));
        cv::Mat trans_mat(3, 3, CV_32FC1, (void *)(trans_matrix + i * 9));
        if(flags & cv::WARP_INVERSE_MAP)
        {
            trans_mat.convertTo(coeffsMat, coeffsMat.type());
        }
        else
        {
            cv::Mat iM;
            cv::invert(trans_mat, iM);
            iM.convertTo(coeffsMat, coeffsMat.type());
        }

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
    float *d_aCoeffs_gpu = (float *)((char *)out_cols_gpu + sizeof(int) * batch);

    checkCudaErrors(cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, buffer_size, cudaMemcpyHostToDevice,
                                    stream));

    typedef void (*func_t)(const void **input, void **output, float *d_coeffs, const int *height, const int *width,
                           const int *out_height, const int *out_width, const int max_height, const int max_width, const int batch,
                           const int interpolation, int borderMode, const float *borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {warpPerspective<uchar>, 0 /*warpPerspective<uchar2>*/, warpPerspective<uchar3>, warpPerspective<uchar4>     },
        {0 /*warpPerspective<schar>*/, 0 /*warpPerspective<char2>*/, 0 /*warpPerspective<char3>*/, 0 /*warpPerspective<char4>*/},
        {warpPerspective<ushort>, 0 /*warpPerspective<ushort2>*/, warpPerspective<ushort3>, warpPerspective<ushort4>    },
        {warpPerspective<short>, 0 /*warpPerspective<short2>*/, warpPerspective<short3>, warpPerspective<short4>     },
        {0 /*warpPerspective<int>*/, 0 /*warpPerspective<int2>*/, 0 /*warpPerspective<int3>*/, 0 /*warpPerspective<int4>*/ },
        {warpPerspective<float>, 0 /*warpPerspective<float2>*/, warpPerspective<float3>, warpPerspective<float4>     }
    };

    const func_t func = funcs[data_type][channels - 1];
    CV_Assert(func != 0);

    cv::Scalar_<float> borderValueFloat;
    borderValueFloat = borderValue;

    func(inputs_gpu, outputs_gpu, d_aCoeffs_gpu, rows_gpu, cols_gpu, out_rows_gpu, out_cols_gpu, max_out_height,
         max_out_width,
         batch, interpolation, borderMode, borderValueFloat.val, stream);
    return 0;
}

} // cuda_op
