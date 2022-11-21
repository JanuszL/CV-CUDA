/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
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

#include "border.cuh"
#include "cuda_utils.cuh"

#define BLOCK 32
using namespace cv::cudev;

template<typename BrdRd, typename T>
__global__ void copyMakeBorderStackKernel(const BrdRd src, cuda_op::Ptr2dNHWC<T> dst, const int *left_, const int *top_)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int       left    = left_[batch_idx];
    int       top     = top_[batch_idx];
    const int x_shift = x - left;
    const int y_shift = y - top;

    int out_height = dst.rows, out_width = dst.cols;

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, y, x) = src(batch_idx, y_shift, x_shift);
    }
}

template<typename BrdRd, typename T>
__global__ void copyMakeBorderKernel(const BrdRd src, cuda_op::Ptr2dVarShapeNHWC<T> dst, const int *left_,
                                     const int *top_)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    int       left    = left_[batch_idx];
    int       top     = top_[batch_idx];
    const int x_shift = x - left;
    const int y_shift = y - top;

    int out_height = dst.rows[batch_idx], out_width = dst.cols[batch_idx];

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, y, x) = src(batch_idx, y_shift, x_shift);
    }
}

template<template<typename> class B, typename T>
struct copyMakeBorderDispatcher
{
    static void call(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst, const T &borderValue,
                     const int *left, const int *top, int max_input_height, int max_input_width, cudaStream_t stream)
    {
        dim3 blockSize(BLOCK, BLOCK / 4, 1);
        dim3 gridSize(divUp(max_input_width, blockSize.x), divUp(max_input_height, blockSize.y), dst.batches);

        B<T>                                                       brd(0, 0, borderValue);
        cuda_op::BorderReader<cuda_op::Ptr2dVarShapeNHWC<T>, B<T>> brdSrc(src, brd);

        copyMakeBorderKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top);
        checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }

    static void callStack(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dNHWC<T> dst, const T &borderValue,
                          const int *left, const int *top, int max_height, int max_width, cudaStream_t stream)
    {
        dim3 blockSize(BLOCK, BLOCK / 4, 1);
        dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), dst.batches);

        B<T>                                                       brd(0, 0, borderValue);
        cuda_op::BorderReader<cuda_op::Ptr2dVarShapeNHWC<T>, B<T>> brdSrc(src, brd);

        copyMakeBorderStackKernel<<<gridSize, blockSize, 0, stream>>>(brdSrc, dst, left, top);
        checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
        checkCudaErrors(cudaStreamSynchronize(stream));
        checkCudaErrors(cudaGetLastError());
#endif
    }
};

template<typename T, int cn> // uchar3 float3 uchar float
void copyMakeBorder(const void **d_in, void **d_out, const int batch_size, const int *height, const int *width,
                    const int *top, const int *left, const int *out_height, const int *out_width, int max_out_height,
                    int max_out_width, bool stack, const int borderType, const cv::Scalar value, cudaStream_t stream)
{
    typedef typename MakeVec<T, cn>::type src_type;
    cv::Scalar_<T>                        value_ = value;
    const src_type                        brdVal = VecTraits<src_type>::make(value_.val);

    cuda_op::Ptr2dVarShapeNHWC<src_type> src_ptr(batch_size, height, width, cn, (src_type **)d_in);

    if (stack)
    {
        cuda_op::Ptr2dNHWC<src_type> dst_ptr(batch_size, max_out_height, max_out_width, cn, (src_type *)d_out[0]);
        typedef void (*func_t)(const cuda_op::Ptr2dVarShapeNHWC<src_type> src, cuda_op::Ptr2dNHWC<src_type> dst,
                               const src_type &borderValue, const int *left, const int *top, int max_height,
                               int max_width, cudaStream_t stream);

        static const func_t funcs[] = {copyMakeBorderDispatcher<cuda_op::BrdConstant, src_type>::callStack,
                                       copyMakeBorderDispatcher<cuda_op::BrdReplicate, src_type>::callStack,
                                       copyMakeBorderDispatcher<cuda_op::BrdReflect, src_type>::callStack,
                                       copyMakeBorderDispatcher<cuda_op::BrdWrap, src_type>::callStack,
                                       copyMakeBorderDispatcher<cuda_op::BrdReflect101, src_type>::callStack};

        funcs[borderType](src_ptr, dst_ptr, brdVal, left, top, max_out_height, max_out_width, stream);
    }
    else
    {
        cuda_op::Ptr2dVarShapeNHWC<src_type> dst_ptr(batch_size, out_height, out_width, cn, (src_type **)d_out);
        typedef void (*func_t)(const cuda_op::Ptr2dVarShapeNHWC<src_type> src, cuda_op::Ptr2dVarShapeNHWC<src_type> dst,
                               const src_type &borderValue, const int *left, const int *top, int max_height,
                               int max_width, cudaStream_t stream);

        static const func_t funcs[] = {copyMakeBorderDispatcher<cuda_op::BrdConstant, src_type>::call,
                                       copyMakeBorderDispatcher<cuda_op::BrdReplicate, src_type>::call,
                                       copyMakeBorderDispatcher<cuda_op::BrdReflect, src_type>::call,
                                       copyMakeBorderDispatcher<cuda_op::BrdWrap, src_type>::call,
                                       copyMakeBorderDispatcher<cuda_op::BrdReflect101, src_type>::call};

        funcs[borderType](src_ptr, dst_ptr, brdVal, left, top, max_out_height, max_out_width, stream);
    }
}

namespace cuda_op {

size_t CopyMakeBorderVarShape::calBufferSize(int batch_size)
{
    // calculate the cpu buffer size for batch of gpu_ptr, height, width, out_height, out_width, top and left
    return (sizeof(void *) * 2 + sizeof(int) * 6) * batch_size;
}

int CopyMakeBorderVarShape::infer(const void **data_in, void **data_out, void *gpu_workspace, void *cpu_workspace,
                                  const int batch, const size_t buffer_size, const int *top, const int *bottom,
                                  const int *left, const int *right, bool stack, const int borderType,
                                  const cv::Scalar value, DataShape *input_shape, DataFormat format, DataType data_type,
                                  cudaStream_t stream)
{
    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = input_shape[0].C;
    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    if (!(borderType == cv::BORDER_CONSTANT || borderType == cv::BORDER_REPLICATE || borderType == cv::BORDER_REFLECT
          || borderType == cv::BORDER_REFLECT_101 || borderType == cv::BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderType " << borderType);
        return ErrorCode::INVALID_PARAMETER;
    }

    size_t data_size = DataSize(data_type);

    const void **inputs_cpu     = (const void **)cpu_workspace;
    void       **outputs_cpu    = (void **)((char *)inputs_cpu + sizeof(void *) * batch);
    int         *top_cpu        = (int *)((char *)outputs_cpu + sizeof(void *) * batch);
    int         *left_cpu       = (int *)((char *)top_cpu + sizeof(int) * batch);
    int         *rows_cpu       = (int *)((char *)left_cpu + sizeof(int) * batch);
    int         *cols_cpu       = (int *)((char *)rows_cpu + sizeof(int) * batch);
    int         *out_rows_cpu   = (int *)((char *)cols_cpu + sizeof(int) * batch);
    int         *out_cols_cpu   = (int *)((char *)out_rows_cpu + sizeof(int) * batch);
    int          max_out_height = 0, max_out_width = 0;

    for (int i = 0; i < batch; i++)
    {
        inputs_cpu[i] = data_in[i];
        if (!stack)
        {
            outputs_cpu[i] = data_out[i];
        }
        else
        {
            outputs_cpu[i] = data_out[0];
        }
        top_cpu[i]      = top[i];
        left_cpu[i]     = left[i];
        rows_cpu[i]     = input_shape[i].H;
        cols_cpu[i]     = input_shape[i].W;
        out_rows_cpu[i] = top[i] + bottom[i] + input_shape[i].H;
        out_cols_cpu[i] = left[i] + right[i] + input_shape[i].W;
        if (stack && (out_rows_cpu[i] != out_rows_cpu[0] || out_cols_cpu[i] != out_cols_cpu[0]))
        {
            LOG_ERROR("Invalid DataShape");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        if (out_cols_cpu[i] > max_out_width)
            max_out_width = out_cols_cpu[i];
        if (out_rows_cpu[i] > max_out_height)
            max_out_height = out_rows_cpu[i];
    }

    const void **inputs_gpu   = (const void **)gpu_workspace;
    void       **outputs_gpu  = (void **)((char *)inputs_gpu + sizeof(void *) * batch);
    int         *top_gpu      = (int *)((char *)outputs_gpu + sizeof(void *) * batch);
    int         *left_gpu     = (int *)((char *)top_gpu + sizeof(int) * batch);
    int         *rows_gpu     = (int *)((char *)left_gpu + sizeof(int) * batch);
    int         *cols_gpu     = (int *)((char *)rows_gpu + sizeof(int) * batch);
    int         *out_rows_gpu = (int *)((char *)cols_gpu + sizeof(int) * batch);
    int         *out_cols_gpu = (int *)((char *)out_rows_gpu + sizeof(int) * batch);

    checkCudaErrors(
        cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, buffer_size, cudaMemcpyHostToDevice, stream));

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

    typedef void (*func_t)(const void **d_in, void **d_out, const int batch_size, const int *height, const int *width,
                           const int *top, const int *left, const int *out_height, const int *out_width,
                           int max_out_height, int max_out_width, bool stack, const int borderType,
                           const cv::Scalar value, cudaStream_t stream);

    // clang-format off
    static const func_t funcs[6][4] =
    {
        {copyMakeBorder<uchar, 1>, copyMakeBorder<uchar, 2>, copyMakeBorder<uchar, 3>, copyMakeBorder<uchar, 4>  },
        {0 /*copyMakeBorder<schar , 1>*/, 0 /*copyMakeBorder<schar , 2>*/, 0 /*copyMakeBorder<schar , 3>*/, 0 /*copyMakeBorder<schar , 4>*/},
        {copyMakeBorder<ushort, 1>, 0 /*copyMakeBorder<ushort, 2>*/, copyMakeBorder<ushort, 3>, copyMakeBorder<ushort, 4>  },
        {copyMakeBorder<short, 1>, 0 /*copyMakeBorder<short , 2>*/, copyMakeBorder<short, 3>, copyMakeBorder<short, 4>  },
        {0 /*copyMakeBorder<int   , 1>*/, 0 /*copyMakeBorder<int   , 2>*/, 0 /*copyMakeBorder<int   , 3>*/, 0 /*copyMakeBorder<int   , 4>*/},
        {copyMakeBorder<float, 1>, 0 /*copyMakeBorder<float , 2>*/, copyMakeBorder<float, 3>, copyMakeBorder<float, 4>  }
    };
    // clang-format on

    const func_t func = funcs[data_type][channels - 1];

    if (stack)
    {
        func(inputs_gpu, outputs_cpu, batch, rows_gpu, cols_gpu, top_gpu, left_gpu, out_rows_gpu, out_cols_gpu,
             max_out_height, max_out_width, stack, borderType, value, stream);
    }
    else
    {
        func(inputs_gpu, outputs_gpu, batch, rows_gpu, cols_gpu, top_gpu, left_gpu, out_rows_gpu, out_cols_gpu,
             max_out_height, max_out_width, stack, borderType, value, stream);
    }
    return 0;
}

} // namespace cuda_op
