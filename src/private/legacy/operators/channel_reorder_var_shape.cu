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

template<typename T>
__global__ void channel_reorder_kernel(const cuda_op::Ptr2dVarShapeNHWC<T> src, cuda_op::Ptr2dVarShapeNHWC<T> dst,
                                       const int *orders)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows[batch_idx], out_width = dst.cols[batch_idx];
    if (dst_x >= out_width || dst_y >= out_height)
        return;
    int order_idx = dst.ch * batch_idx;
    for (int ch = 0; ch < dst.ch; ch++)
    {
        int src_ch = orders[order_idx + ch];
        if (src_ch < 0)
        {
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = 0;
        }
        else
        {
            *dst.ptr(batch_idx, dst_y, dst_x, ch) = *src.ptr(batch_idx, dst_y, dst_x, src_ch);
        }
    }
}

template<typename T>
void reorder(const void **d_in, void **d_out, const int batch_size, const int *height, const int *width,
             const int max_height, const int max_width, int in_channels, int out_channels, const int *orders,
             cudaStream_t stream)
{
    dim3 blockSize(BLOCK, BLOCK / 4, 1);
    dim3 gridSize(divUp(max_width, blockSize.x), divUp(max_height, blockSize.y), batch_size);

    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(batch_size, height, width, in_channels, (T **)d_in);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(batch_size, height, width, out_channels, (T **)d_out);

    channel_reorder_kernel<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, orders);
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace cuda_op {

size_t ChannelReorderVarShape::calBufferSize(int batch_size, int channel_size)
{
    return (sizeof(void *) * 2 + sizeof(int) * (2 + channel_size)) * batch_size;
}

int ChannelReorderVarShape::infer(const void **data_in, void **data_out, void *gpu_workspace, void *cpu_workspace,
                                  const int batch, const size_t buffer_size, const int *orders_in, int out_channels,
                                  DataShape *input_shape, DataFormat format, DataType data_type, cudaStream_t stream)
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

    if (out_channels > 4)
    {
        printf("Invalid channel number %d\n", out_channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const void **inputs  = (const void **)cpu_workspace;
    void       **outputs = (void **)((char *)inputs + sizeof(void *) * batch);
    int         *rows    = (int *)((char *)outputs + sizeof(void *) * batch);
    int         *cols    = (int *)((char *)rows + sizeof(int) * batch);
    int         *orders  = (int *)((char *)cols + sizeof(int) * batch);

    int max_height = 0, max_width = 0;

    for (int i = 0; i < batch; ++i)
    {
        if (channels != input_shape[i].C)
        {
            LOG_ERROR("Invalid Input");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        inputs[i]  = data_in[i];
        outputs[i] = data_out[i];
        rows[i]    = input_shape[i].H;
        cols[i]    = input_shape[i].W;
        for (int ch = 0; ch < out_channels; ch++)
        {
            int tmp_idx = out_channels * i + ch;
            if (orders_in[tmp_idx] >= channels)
            {
                printf("Invliad channel order %d\n", orders_in[tmp_idx]);
                return ErrorCode::INVALID_DATA_SHAPE;
            }
            orders[tmp_idx] = orders_in[tmp_idx];
        }

        if (cols[i] > max_width)
            max_width = cols[i];
        if (rows[i] > max_height)
            max_height = rows[i];
    }

    const void **inputs_gpu  = (const void **)gpu_workspace;
    void       **outputs_gpu = (void **)((char *)inputs_gpu + sizeof(void *) * batch);
    int         *rows_gpu    = (int *)((char *)outputs_gpu + sizeof(void *) * batch);
    int         *cols_gpu    = (int *)((char *)rows_gpu + sizeof(int) * batch);
    int         *orders_gpu  = (int *)((char *)cols_gpu + sizeof(int) * batch);

    checkCudaErrors(
        cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, buffer_size, cudaMemcpyHostToDevice, stream));

    typedef void (*func_t)(const void **d_in, void **d_out, const int batch_size, const int *height, const int *width,
                           const int max_width, const int max_height, int in_channels, int out_channels,
                           const int *orders, cudaStream_t stream);

    static const func_t funcs[6] = {reorder<uchar>, 0, reorder<ushort>, reorder<short>, reorder<int>, reorder<float>};

    const func_t func = funcs[data_type];
    CV_Assert(func != 0);

    func(inputs_gpu, outputs_gpu, batch, rows_gpu, cols_gpu, max_height, max_width, channels, out_channels, orders_gpu,
         stream);
    return 0;
}

} // namespace cuda_op
