/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Copyright (c) 2021-2022, Bytedance Inc. All rights reserved.
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

#ifndef CV_CUDA_UTILS_H_
#define CV_CUDA_UTILS_H_

#include <cmath>
#include <cstdio>

namespace cuda_op {

typedef unsigned char uchar;
typedef char          schar;

#define get_batch_idx() (blockIdx.z)

int divUp(int a, int b)
{
    return ceil((float)a / b);
};

template<typename T>
struct Ptr2dNCHW
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dNCHW()
        : batches(0)
        , rows(0)
        , cols(0)
        , ch(0)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNCHW(int rows_, int cols_, int ch_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNCHW(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , data(data_)
    {
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        return (T *)(data + b * ch * rows * cols + c * rows * cols + y * cols + x);
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        return (const T *)(data + b * ch * rows * cols + c * rows * cols + y * cols + x);
    }

    int batches;
    int rows;
    int cols;
    int ch;
    T  *data;
};

template<typename T>
struct Ptr2dNHWC
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dNHWC()
        : batches(0)
        , rows(0)
        , cols(0)
        , ch(0)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int rows_, int cols_, int ch_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , data(data_)
    {
    }

    // ptr for uchar1/3/4, ushort1/3/4, float1/3/4, typename T -> uchar3 etc.
    // each fetch operation get a x-channel elements
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x)
    {
        return (T *)(data + b * rows * cols + y * cols + x);
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x) const
    {
        return (const T *)(data + b * rows * cols + y * cols + x);
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        return (T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        return (const T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
    }

    __host__ __device__ __forceinline__ int at_rows(int b)
    {
        return rows;
    }

    __host__ __device__ __forceinline__ int at_rows(int b) const
    {
        return rows;
    }

    __host__ __device__ __forceinline__ int at_cols(int b)
    {
        return cols;
    }

    __host__ __device__ __forceinline__ int at_cols(int b) const
    {
        return cols;
    }

    int batches;
    int rows;
    int cols;
    int ch;
    T  *data;
};

#ifndef checkKernelErrors
#    define checkKernelErrors(expr)                                                               \
        do                                                                                        \
        {                                                                                         \
            expr;                                                                                 \
                                                                                                  \
            cudaError_t __err = cudaGetLastError();                                               \
            if (__err != cudaSuccess)                                                             \
            {                                                                                     \
                printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
                abort();                                                                          \
            }                                                                                     \
        }                                                                                         \
        while (0)
#endif

#ifndef checkCudaErrors
#    define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        const char *errorStr = NULL;
        errorStr             = cudaGetErrorString(err);
        fprintf(stderr,
                "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
                "line %i.\n",
                err, errorStr, file, line);
        exit(1);
    }
}
#endif

} // namespace cuda_op
#endif
