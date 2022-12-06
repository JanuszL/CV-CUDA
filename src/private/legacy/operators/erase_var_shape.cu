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
#include "cv_utils.h"

#include "border.cuh"
#include "cuda_utils.cuh"

#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace cv::cudev;
using namespace cuda_op;

__device__ int erase_var_shape_hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

template<typename D, typename Ptr2D>
__global__ void erase(Ptr2D img, int *gpu_rows, int *gpu_cols, int *gpu_anchor_x, int *gpu_anchor_y, int *gpu_erasing_w,
                      int *gpu_erasing_h, int *gpu_erasing_c, float *gpu_values, int *gpu_imgIdx, int channels,
                      int random, unsigned int seed)
{
    unsigned int id        = threadIdx.x + blockIdx.x * blockDim.x;
    int          c         = blockIdx.y;
    int          eraseId   = blockIdx.z;
    int          anchor_x  = gpu_anchor_x[eraseId];
    int          anchor_y  = gpu_anchor_y[eraseId];
    int          erasing_w = gpu_erasing_w[eraseId];
    int          erasing_h = gpu_erasing_h[eraseId];
    int          erasing_c = gpu_erasing_c[eraseId];
    float        value     = gpu_values[eraseId * channels + c];
    int          batchId   = gpu_imgIdx[eraseId];
    if (id < erasing_h * erasing_w && (0x1 & (erasing_c >> c)) == 1)
    {
        int x = id % erasing_w;
        int y = id / erasing_w;
        if ((anchor_x + x) < gpu_cols[batchId] && (anchor_y + y) < gpu_rows[batchId])
        {
            if (random)
            {
                unsigned int hashValue = seed + threadIdx.x
                                       + 0x26AD0C9 * blockDim.x * blockDim.y * blockDim.z * (blockIdx.x + 1)
                                             * (blockIdx.y + 1) * (blockIdx.z + 1);
                *img.ptr(batchId, anchor_y + y, anchor_x + x, c)
                    = saturate_cast<D>(erase_var_shape_hash(hashValue) % 256);
            }
            else
            {
                *img.ptr(batchId, anchor_y + y, anchor_x + x, c) = saturate_cast<D>(value);
            }
        }
    }
}

template<typename D>
void eraseCaller(void **imgs, int *gpu_rows, int *gpu_cols, int batch, int channel, int *gpu_anchor_x,
                 int *gpu_anchor_y, int *gpu_erasing_w, int *gpu_erasing_h, int *gpu_erasing_c, float *gpu_values,
                 int *gpu_imgIdx, int max_eh, int max_ew, int num_erasing_area, bool random, unsigned int seed,
                 cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<D> src(batch, gpu_rows, gpu_cols, channel, (D **)imgs);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    int  blockSize = (max_eh * max_ew < 1024) ? max_eh * max_ew : 1024;
    int  gridSize  = divUp(max_eh * max_ew, 1024);
    dim3 block(blockSize);
    dim3 grid(gridSize, channel, num_erasing_area);
    erase<D, Ptr2dVarShapeNHWC<D>><<<grid, block, 0, stream>>>(src, gpu_rows, gpu_cols, gpu_anchor_x, gpu_anchor_y,
                                                               gpu_erasing_w, gpu_erasing_h, gpu_erasing_c, gpu_values,
                                                               gpu_imgIdx, channel, random, seed);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace cuda_op {

size_t EraseVarShape::calBufferSize(int batch_size, int num_erasing_area)
{
    return num_erasing_area * (sizeof(int) * 6 + sizeof(float) * 4)
         + (2 * sizeof(void *) + 2 * sizeof(int)) * batch_size;
}

// todo support random value && rgb value

int EraseVarShape::infer(void **inputs, void **outputs, void *gpu_workspace, void *cpu_workspace, const int batch,
                         const size_t buffer_size, int *anchor_x, int *anchor_y, int *erasing_w, int *erasing_h,
                         int *erasing_c, float *values, int *imgIdx, int num_erasing_area, bool random,
                         unsigned int seed, DataShape *input_shapes, DataFormat format, DataType data_type,
                         cudaStream_t stream, bool inplace)
{
    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (num_erasing_area < 0)
    {
        LOG_ERROR("Invalid num of erasing area " << num_erasing_area);
        return ErrorCode::INVALID_PARAMETER;
    }

    void **gpu_inputs    = (void **)gpu_workspace;
    void **gpu_outputs   = (void **)((char *)gpu_inputs + sizeof(void *) * batch);
    int   *gpu_rows      = (int *)((char *)gpu_outputs + sizeof(void *) * batch);
    int   *gpu_cols      = (int *)((char *)gpu_rows + sizeof(int) * batch);
    int   *gpu_anchor_x  = gpu_cols + batch;
    int   *gpu_anchor_y  = gpu_anchor_x + num_erasing_area;
    int   *gpu_erasing_w = gpu_anchor_y + num_erasing_area;
    int   *gpu_erasing_h = gpu_erasing_w + num_erasing_area;
    int   *gpu_erasing_c = gpu_erasing_h + num_erasing_area;
    int   *gpu_imgIdx    = gpu_erasing_c + num_erasing_area;
    float *gpu_values    = (float *)(gpu_imgIdx + num_erasing_area);

    void **cpu_inputs    = (void **)cpu_workspace;
    void **cpu_outputs   = (void **)((char *)cpu_inputs + sizeof(void *) * batch);
    int   *cpu_rows      = (int *)((char *)cpu_outputs + sizeof(void *) * batch);
    int   *cpu_cols      = (int *)((char *)cpu_rows + sizeof(int) * batch);
    int   *cpu_anchor_x  = cpu_cols + batch;
    int   *cpu_anchor_y  = cpu_anchor_x + num_erasing_area;
    int   *cpu_erasing_w = cpu_anchor_y + num_erasing_area;
    int   *cpu_erasing_h = cpu_erasing_w + num_erasing_area;
    int   *cpu_erasing_c = cpu_erasing_h + num_erasing_area;
    int   *cpu_imgIdx    = cpu_erasing_c + num_erasing_area;
    float *cpu_values    = (float *)(cpu_imgIdx + num_erasing_area);

    const int channels = input_shapes[0].C;

    for (int i = 0; i < batch; i++)
    {
        if (channels != input_shapes[i].C)
        {
            LOG_ERROR("Invalid Input " << input_shapes[i].C);
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        cpu_rows[i] = input_shapes[i].H;
        cpu_cols[i] = input_shapes[i].W;

        if (!inplace)
        {
            checkCudaErrors(cudaMemcpyAsync(
                outputs[i], inputs[i], DataSize(data_type) * input_shapes[i].C * input_shapes[i].H * input_shapes[i].W,
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    if (num_erasing_area == 0)
    {
        return 0;
    }

    std::memcpy(cpu_inputs, inputs, sizeof(void *) * batch);
    std::memcpy(cpu_outputs, outputs, sizeof(void *) * batch);
    std::memcpy(cpu_anchor_x, anchor_x, sizeof(int) * num_erasing_area);
    std::memcpy(cpu_anchor_y, anchor_y, sizeof(int) * num_erasing_area);
    std::memcpy(cpu_erasing_w, erasing_w, sizeof(int) * num_erasing_area);
    std::memcpy(cpu_erasing_h, erasing_h, sizeof(int) * num_erasing_area);
    std::memcpy(cpu_erasing_c, erasing_c, sizeof(int) * num_erasing_area);
    std::memcpy(cpu_imgIdx, imgIdx, sizeof(int) * num_erasing_area);

    if (random)
    {
        checkCudaErrors(
            cudaMemcpyAsync(gpu_workspace, cpu_workspace,
                            num_erasing_area * sizeof(int) * 6 + (2 * sizeof(void *) + 2 * sizeof(int)) * batch,
                            cudaMemcpyHostToDevice, stream));
    }
    else
    {
        std::memcpy(cpu_values, values, sizeof(float) * num_erasing_area * 4);
        checkCudaErrors(cudaMemcpyAsync(
            gpu_workspace, cpu_workspace,
            num_erasing_area * (sizeof(int) * 6 + sizeof(float) * 4) + (2 * sizeof(void *) + 2 * sizeof(int)) * batch,
            cudaMemcpyHostToDevice, stream));
    }
    void **inputImgs;
    if (inplace)
    {
        inputImgs = gpu_inputs;
    }
    else
    {
        inputImgs = gpu_outputs;
    }

    int max_eh = 0, max_ew = 0;
    for (int i = 0; i < num_erasing_area; i++)
    {
        int eh = erasing_h[i], ew = erasing_w[i];
        if (eh * ew > max_eh * max_ew)
        {
            max_eh = eh;
            max_ew = ew;
        }
    }

    typedef void (*erase_t)(void **imgs, int *gpu_rows, int *gpu_cols, int batch, int channel, int *gpu_anchor_x,
                            int *gpu_anchor_y, int *gpu_erasing_w, int *gpu_erasing_h, int *gpu_erasing_c,
                            float *gpu_values, int *gpu_imgIdx, int max_eh, int max_ew, int num_erasing_area,
                            bool random, unsigned int seed, cudaStream_t stream);

    static const erase_t funcs[6] = {eraseCaller<uchar>, eraseCaller<char>, eraseCaller<ushort>,
                                     eraseCaller<short>, eraseCaller<int>,  eraseCaller<float>};

    funcs[data_type](inputImgs, gpu_rows, gpu_cols, batch, channels, gpu_anchor_x, gpu_anchor_y, gpu_erasing_w,
                     gpu_erasing_h, gpu_erasing_c, gpu_values, gpu_imgIdx, max_eh, max_ew, num_erasing_area, random,
                     seed, stream);

    return 0;
}

} // namespace cuda_op*/
