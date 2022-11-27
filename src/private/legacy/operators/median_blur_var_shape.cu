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
#include "cv_utils.h"
#include "helper_math.h"
#include "curand.h"
#include "curand_kernel.h"


#define GENERAL_KERNEL_BLOCK 32
#define SMALL_KERNEL_BLOCK 16
using namespace cv::cudev;
using namespace cuda_op;

/**
 * This function fetches the pixel from the shared if possible.
 * Otherwise, the pixel is read from global memory.
 * If the given index is out of bound, then based on the rule of cv::BORDER_REPLICATE,
 * this function fetches the nearest valid pixel.
 * @tparam T The type of the pixels stored.
 * @param shared a pointer of type T to shared memory,
 * @param src a Ptr2dNHWC <T> stored in global memory.
 * @param batchIdx the index of the image.
 * @param h the height of the image.
 * @param w the width of the image.
 * @param c the channel being processed.
 * @param sxOffset the x offset that is subtracted from gx to calculate the corresponding pixel index in shared memory.
 * @param syOffset the y offset that is subtracted from gy to calculate the corresponding pixel index in shared memory.
 * @param gx the horizontal index of the desired pixel in the image.
 * @param gy the vertical index of the desired pixel in the image.
 * @return the pixel at given index
 */
template<typename T>
__device__ T fetch(T *shared, const Ptr2dVarShapeNHWC <T> src, int batchIdx, int h, int w, int c,
                   int sxOffset, int syOffset, int gx, int gy, int block_size)
{
    // check for cv::BORDER_REPLICATE.
    if(gx < 0)
    {
        gx = 0;
    }
    if(gx >= w)
    {
        gx = w - 1;
    }
    if(gy < 0)
    {
        gy = 0;
    }
    if(gy >= h)
    {
        gy = h - 1;
    }
    // check if the desired pixel is not in shared memory.
    if(gy - syOffset < 0 || gy - syOffset >= blockDim.y ||
            gx - sxOffset < 0 || gx - sxOffset >= blockDim.x)
    {
        return *src.ptr(batchIdx, gy, gx, c); // fetch from global memory.
    }
    else
    {
        return shared[(gy - syOffset) * block_size + gx - sxOffset];  // fetch from shared memory.
    }
}


/**
 * Perform median fliter on the image
 * @tparam T The type of the pixels stored.
 * @param src a Ptr2dNHWC <T> stored in global memory.
 * @param dst a Ptr2dNHWC <T> stored in global memory.
 * @param kWidth width of the kernel.
 * @param kHeight height of the kernel.
 */
template<typename T>
__global__ void median(const Ptr2dVarShapeNHWC <T> src, Ptr2dVarShapeNHWC <T> dst,
                       const int *kWidths, const int *kHeights)
{
#define fetch_(gx, gy, block_size) fetch<T>(tails, src, batchIdx, h, w, channel, blockX, blockY, (gx), (gy), (block_size))
#define fetchAs1d(idx, block_size) fetch_(x - (kWidth / 2) + ((idx) % kWidth),\
        y - (kHeight / 2) + ((idx) / kWidth), (block_size))
    int tx = threadIdx.x, ty = threadIdx.y;
    int blockX = blockIdx.x * blockDim.x;
    int blockY = blockIdx.y * blockDim.y;
    int x = blockX + threadIdx.x;
    int y = blockY + threadIdx.y;
    int channel = blockIdx.z % dst.ch;
    int batchIdx = blockIdx.z / dst.ch;
    int h = src.rows[batchIdx], w = src.cols[batchIdx];
    int kWidth = kWidths[batchIdx];
    int kHeight = kHeights[batchIdx];

    __shared__ T tails[GENERAL_KERNEL_BLOCK * GENERAL_KERNEL_BLOCK];
    if(x < w && y < h)
    {
        tails[ty * GENERAL_KERNEL_BLOCK + tx] = *src.ptr(batchIdx, y, x, channel);
    }
    __syncthreads();

    if((x < w && y < h))
    {
        // min_ and max_ set up a range that we are looking for
        // only elements in that range could be median
        T tmp, pivot0, pivot1, pivot2, min_, max_;
        // In the 1st and possibly several following iterations, min_ or max_ is not assigned.
        // use isMinReady and isMaxReady to control from comparison on them.
        bool isMinReady = false, isMaxReady = false;
        int numOfEq = 0, numOfGt = 0, numOfLt = 0, numOfTaken = 0;
        int median = (kWidth * kHeight) / 2;
        int start = 0, end = kWidth * kHeight, t;
        bool isAllPreviousOutOfRange = true;

        // loop until we rule out all possible elements, and the last pivot is the median.
        while(numOfTaken < (kWidth * kHeight))
        {
            pivot0 = fetchAs1d(start, GENERAL_KERNEL_BLOCK);
            while((isMinReady && (min_ >= pivot0)) ||
                    (isMaxReady && (max_ <= pivot0)))
            {
                start++;
                pivot0 = fetchAs1d(start, GENERAL_KERNEL_BLOCK);
            }

            pivot2 = fetchAs1d(end - 1, GENERAL_KERNEL_BLOCK);
            while((isMinReady && (min_ >= pivot2)) ||
                    (isMaxReady && (max_ <= pivot2)))
            {
                end--;
                pivot2 = fetchAs1d(end - 1, GENERAL_KERNEL_BLOCK);
            }

            int idx = (start + end) / 2;
            pivot1 = fetchAs1d(idx, GENERAL_KERNEL_BLOCK);
            // check if the pivot is in the range defined by min_ and max_.
            // if not, go to the next until we find one that is in the range.
            while((isMinReady && (min_ >= pivot1)) ||
                    (isMaxReady && (max_ <= pivot1)))
            {
                idx++;
                if(idx >= end)
                {
                    idx = start;
                }
                pivot1 = fetchAs1d(idx, GENERAL_KERNEL_BLOCK);
            }

            if(pivot0 < pivot1 && pivot1 < pivot2)
            {
                pivot0 = pivot1;
            }
            else if(pivot0 < pivot2 && pivot2 < pivot1)
            {
                pivot0 = pivot2;
            }

            // use the pivot to partition the array.
            t = end;
            for(int i = start; i < t; i++)
            {
                tmp = fetchAs1d(i, GENERAL_KERNEL_BLOCK);
                // only consider the element in the range defined by min_ and max_.
                // because others are already ruled out.
                if((!isMinReady || min_ < tmp) && (!isMaxReady || tmp < max_))
                {
                    if(tmp > pivot0)
                    {
                        numOfGt++;
                    }
                    else if(tmp < pivot0)
                    {
                        numOfLt++;
                    }
                    else
                    {
                        numOfEq++;
                    }
                    if(isAllPreviousOutOfRange)
                    {
                        start = i;
                        isAllPreviousOutOfRange = false;
                    }
                    end = i + 1;
                }
            }

            // if the index of median is less than numOfLt,
            // use max_ to rule out elements greater than or equal to pivot.
            if(median < numOfLt)
            {
                max_ = pivot0;
                numOfTaken = numOfTaken + numOfEq + numOfGt;
                isMaxReady = true;
                // if the index of median is in between numOfLt and (numOfLt + numOfEq).
                // the median is found. we are lucky:)
            }
            else if(median < (numOfLt + numOfEq))
            {
                break;
                // if the index of median is greater than (numOfLt + numOfEq),
                // use min_ to rule out elements greater than or equal to pivot.
            }
            else
            {
                min_ = pivot0;
                median = median - numOfLt - numOfEq;
                numOfTaken = numOfTaken + numOfLt + numOfEq;
                isMinReady = true;
            }
            numOfLt = 0;
            numOfEq = 0;
            numOfGt = 0;
        }
        *dst.ptr(batchIdx, y, x, channel) = pivot0;
    }
}

template<typename T>
__device__ int partition(T *arr, int length, T pvt, int *numOfEq)
{
    T val;
    *numOfEq = 1;
    int i = 1;
    for(int j = 1; j < length; j++)
    {
        val = arr[j];
        if(val == pvt)
        {
            (*numOfEq) += 1;
        }
        if(val < pvt)
        {
            arr[j] = arr[i];
            arr[i] = val;
            i += 1;
        }
    }
    val = arr[0];
    arr[0] = arr[i - 1];
    arr[i - 1] = val;
    return i - 1;
}

template<typename T>
__inline__ __device__ T placePivot(T *arr, int length)
{
    int mid = length / 2;
    T pivot0 = arr[0], pivot1 = arr[mid], pivot2 = arr[length - 1];
    if(pivot0 < pivot1 && pivot1 <= pivot2)
    {
        arr[0] = pivot1;
        arr[mid] = pivot0;
        return pivot1;
    }
    if(pivot0 < pivot2 && pivot2 <= pivot1)
    {
        arr[0] = pivot2;
        arr[length - 1] = pivot0;
        return pivot2;
    }
    return pivot0;
}

template<typename T>
__global__ void medianForSmallKernel(const Ptr2dVarShapeNHWC <T> src, Ptr2dVarShapeNHWC <T> dst,
                                     const int *kWidths, const int *kHeights)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int blockX = blockIdx.x * blockDim.x;
    int blockY = blockIdx.y * blockDim.y;
    int x = blockX + threadIdx.x;
    int y = blockY + threadIdx.y;
    int channel = blockIdx.z % dst.ch;
    int batchIdx = blockIdx.z / dst.ch;
    int h = src.rows[batchIdx], w = src.cols[batchIdx];
    int kWidth = kWidths[batchIdx];
    int kHeight = kHeights[batchIdx];

    __shared__ T tails[SMALL_KERNEL_BLOCK * SMALL_KERNEL_BLOCK];
    if(x < w && y < h)
    {
        tails[ty * SMALL_KERNEL_BLOCK + tx] = *src.ptr(batchIdx, y, x, channel);
    }
    __syncthreads();

    extern  __shared__ char _arrays[];
    int length = kWidth * kHeight;
    T *arr = ((T *) _arrays) + ((tx * SMALL_KERNEL_BLOCK) + ty) * length;
    T pivot;
    int numOfEq, k = length / 2;

    if((x < w && y < h))
    {
        for(int i = 0; i < length; i++)
        {
            arr[i] = fetchAs1d(i, SMALL_KERNEL_BLOCK);
        }
        while(length > 1)
        {
            pivot = placePivot(arr, length);
            int middle = partition(arr, length, pivot, &numOfEq);
            if(k < middle)
            {
                length = middle;
            }
            else if(k < (middle + numOfEq))
            {
                *dst.ptr(batchIdx, y, x, channel) = pivot;
                return;
            }
            else
            {
                k = k - middle - 1;
                length = length - middle - 1;
                arr = arr + middle + 1;
            }
        }
        *dst.ptr(batchIdx, y, x, channel) = arr[0];
    }
}


#undef fetch_
#undef fetchAs1d

template<typename T>
void median(const void **input, void **output, const int *height,
            const int *width, int max_height, int max_width, int batch, int channels,
            int *kWidths, int *kHeights, int maxKWidth, int maxKHeight, cudaStream_t stream)
{
    Ptr2dVarShapeNHWC <T> src(batch, height, width, channels, (T **) input);
    Ptr2dVarShapeNHWC <T> dst(batch, height, width, channels, (T **) output);

    #ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
    #endif

    long unsigned int sharedMemSize = SMALL_KERNEL_BLOCK * SMALL_KERNEL_BLOCK * maxKWidth * maxKHeight * sizeof(T);
    if(sharedMemSize < 48 * 1024)
    {
        dim3 block(SMALL_KERNEL_BLOCK, SMALL_KERNEL_BLOCK);
        dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), dst.ch * dst.batches);
        medianForSmallKernel<T><<<grid, block, sharedMemSize, stream>>>(src, dst, kWidths, kHeights);
        checkKernelErrors();
    }
    else
    {
        dim3 block(GENERAL_KERNEL_BLOCK, GENERAL_KERNEL_BLOCK);
        dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), dst.ch * dst.batches);
        median<T><<<grid, block, 0, stream>>>(src, dst, kWidths, kHeights);
        checkKernelErrors();
    }

    #ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
    #endif
}

namespace cuda_op
{

size_t MedianBlurVarShape::calBufferSize(int batch_size)
{
    return (2 * sizeof(void *) + 4 * sizeof(int)) * batch_size;
}

int MedianBlurVarShape::infer(const void **data_in, void **data_out, void *gpu_workspace, void *cpu_workspace,
                              const int batch, const size_t buffer_size, cv::Size *ksize, DataShape *inputShape,
                              DataFormat format, DataType data_type, cudaStream_t stream)
{
    if(!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if(!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inputShape[0].C;

    if(channels > 4)
    {
        printf("Invalid channel number %d\n", channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const void **inputs = (const void **) cpu_workspace;
    void **outputs = (void **)((char *) inputs + sizeof(void *) * batch);
    int *rows = (int *)((char *) outputs + sizeof(void *) * batch);
    int *cols = (int *)((char *) rows + sizeof(int) * batch);
    int *k_width = (int *)((char *) cols + sizeof(int) * batch);
    int *k_height = (int *)((char *) k_width + sizeof(int) * batch);

    size_t data_size = DataSize(data_type);
    int max_height = 0, max_width = 0, max_kHeight = 0, max_kWidth = 0;

    for(int b = 0; b < batch; b++)
    {
        if(channels != inputShape[b].C)
        {
            LOG_ERROR("Invalid Input");
            return ErrorCode::INVALID_DATA_SHAPE;
        }
        inputs[b] = data_in[b];
        outputs[b] = data_out[b];
        rows[b] = inputShape[b].H;
        cols[b] = inputShape[b].W;

        if(!(ksize[b].width > 0 && ksize[b].width % 2 == 1 && ksize[b].height > 0 && ksize[b].height % 2 == 1))
        {
            LOG_ERROR("Invalid ksize " << ksize[b].width << " " << ksize[b].height);
            return ErrorCode::INVALID_PARAMETER;
        }

        k_width[b] = ksize[b].width;
        k_height[b] = ksize[b].height;

        if(cols[b] > max_width)
            max_width = cols[b];
        if(rows[b] > max_height)
            max_height = rows[b];
        if(k_width[b]>max_kWidth)
            max_kWidth = k_width[b];
        if(k_height[b]>max_kHeight)
            max_kHeight = k_height[b];
    }

    const void **inputs_gpu = (const void **) gpu_workspace;
    void **outputs_gpu = (void **)((char *) inputs_gpu + sizeof(void *) * batch);
    int *rows_gpu = (int *)((char *) outputs_gpu + sizeof(void *) * batch);
    int *cols_gpu = (int *)((char *) rows_gpu + sizeof(int) * batch);
    int *k_width_gpu = (int *)((char *) cols_gpu + sizeof(int) * batch);
    int *k_height_gpu = (int *)((char *) k_width_gpu + sizeof(int) * batch);

    checkCudaErrors(
                    cudaMemcpyAsync((void *) gpu_workspace, (void *) cpu_workspace, buffer_size, cudaMemcpyHostToDevice,
                                    stream));

    typedef void (*median_t)(const void **input, void **output, const int *height,
                             const int *width, int max_height, int max_width, int batch,
                             int channels, int *kWidths, int *kHeights, int maxKWidth, int maxKHeight,
                             cudaStream_t stream);

    static const median_t funcs[6] =
    {
        median<uchar>, 0, median<ushort>, 0, median<int>, median<float>,

    };
    funcs[data_type](inputs_gpu, outputs_gpu, rows_gpu, cols_gpu, max_height, max_width,
                     batch, channels, k_width_gpu, k_height_gpu, max_kWidth, max_kHeight, stream);
    return 0;
}

} // cuda_op
#undef GENERAL_KERNEL_BLOCK
#undef SMALL_KERNEL_BLOCK
