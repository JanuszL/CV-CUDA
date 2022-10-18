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

#include <nvcv/IImageData.hpp>
#include <nvcv/ITensorData.hpp>
#include <util/Exception.hpp>

#include <cassert>
#include <cmath>
#include <cstdio>

namespace nv::cv::legacy::cuda_op {

typedef unsigned char uchar;
typedef char          schar;

#define get_batch_idx() (blockIdx.z)

inline int divUp(int a, int b)
{
    assert(b > 0);
    return ceil((float)a / b);
};

template<class T>
struct PlaneArray
{
    __host__ __device__ T *operator[](int c) const
    {
        assert(0 <= c && c < 4);
        return planes[c];
    }

    T *planes[4];
};

// convert legacy NCHW packed image representation into plane array
template<class T>
__host__ __device__ PlaneArray<T> ToPlaneArray(int rows, int cols, int channels, void *data)
{
    PlaneArray<T> p = {};
    for (int c = 0; c < channels; ++c)
    {
        p.planes[c] = reinterpret_cast<T *>(data) + c * rows * cols;
    }
    return p;
}

template<class T> // base type
__host__ __device__ int32_t CalcNCHWImagePitchBytes(int rows, int cols, int channels)
{
    return rows * cols * channels * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNCHWRowPitchBytes(int cols, int channels)
{
    return cols * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNHWCImagePitchBytes(int rows, int cols, int channels)
{
    return rows * cols * channels * sizeof(T);
}

template<class T> // base type
__host__ __device__ int32_t CalcNHWCRowPitchBytes(int cols, int channels)
{
    return cols * channels * sizeof(T);
}

// Used to disambiguate between the constructors that accept legacy memory buffers,
// and the ones that accept the new ones. Just pass NewAPI as first parameter.
struct NewAPITag
{
};

constexpr NewAPITag NewAPI = {};

template<typename T>
struct Ptr2dNCHW
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dNCHW()
        : batches(0)
        , rows(0)
        , cols(0)
        , ch(0)
        , imgPitchBytes(0)
        , chPitchBytes(0)
        , data{}
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNCHW(int rows_, int cols_, int ch_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgPitchBytes(0)
        , rowPitchBytes(CalcNCHWRowPitchBytes<T>(cols, ch_))
        , data(data_)
    {
        chPitchBytes = rowPitchBytes * rows_;
    }

    __host__ __device__ __forceinline__ Ptr2dNCHW(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgPitchBytes(CalcNCHWImagePitchBytes<T>(rows_, cols_, ch_))
        , rowPitchBytes(CalcNCHWRowPitchBytes<T>(cols, ch_))
        , data(data_)
    {
        chPitchBytes = rowPitchBytes * rows_;
    }

    __host__ __forceinline__ Ptr2dNCHW(const IImageDataDevicePitch &inData)
        : batches(1)
        , rows(inData.size().h)
        , cols(inData.size().w)
        , ch(inData.format().numPlanes())
        , imgPitchBytes(0)
    {
        if (inData.format().numPlanes() != inData.format().numChannels())
        {
            throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Image must be planar");
        }

        rowPitchBytes = inData.plane(0).pitchBytes;
        chPitchBytes  = rowPitchBytes * inData.plane(0).height;
        data          = reinterpret_cast<T *>(inData.plane(0).buffer);

        for (int i = 0; i < ch; ++i)
        {
            const ImagePlanePitch &plane = inData.plane(i);

            if (i > 0)
            {
                if (plane.pitchBytes != rowPitchBytes)
                {
                    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "All image planes' row pitch must be the same");
                }

                if (plane.buffer != reinterpret_cast<const std::byte *>(data) + rowPitchBytes * plane.height * i)
                {
                    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "All image buffer must be packed");
                }

                if (inData.format().planePixelType(i) != inData.format().planePixelType(0))
                {
                    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                          "All image planes must have the same pixel type");
                }

                if (plane.width != inData.plane(0).width || plane.height != inData.plane(0).height)
                {
                    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "All image planes must have the same size");
                }
            }
        }
    }

    __host__ __forceinline__ Ptr2dNCHW(const ITensorDataPitchDevice &tensor)
    {
        DimsNCHW dims = tensor.dims();
        batches       = dims.n;
        rows          = dims.h;
        cols          = dims.w;
        ch            = dims.c;

        imgPitchBytes = tensor.pitchBytes(0);
        chPitchBytes  = tensor.pitchBytes(1);
        rowPitchBytes = tensor.pitchBytes(2);
        data          = tensor.mem();
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        //return (T *)(data + b * ch * rows * cols + c * rows * cols + y * cols + x);
        return (T *)(reinterpret_cast<std::byte *>(data) + b * imgPitchBytes + c * chPitchBytes + y * rowPitchBytes
                     + x * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        //return (const T *)(data + b * ch * rows * cols + c * rows * cols + y * cols + x);
        return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgPitchBytes + c * chPitchBytes
                           + y * rowPitchBytes + x * sizeof(T));
    }

    int   batches;
    int   rows;
    int   cols;
    int   ch;
    int   imgPitchBytes;
    int   rowPitchBytes;
    int   chPitchBytes;
    void *data;
};

template<typename T>
struct Ptr2dNHWC
{
    typedef T value_type;

    __host__ __device__ __forceinline__ Ptr2dNHWC()
        : batches(0)
        , rows(0)
        , cols(0)
        , imgPitchBytes(0)
        , rowPitchBytes(0)
        , ch(0)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int rows_, int cols_, int ch_, T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgPitchBytes(0)
        , rowPitchBytes(CalcNHWCRowPitchBytes<T>(cols_, ch_))
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(int batches_, int rows_, int cols_, int ch_, T *data_)
        : batches(batches_)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgPitchBytes(CalcNHWCImagePitchBytes<T>(rows_, cols_, ch_))
        , rowPitchBytes(CalcNHWCRowPitchBytes<T>(cols_, ch_))
        , data(data_)
    {
    }

    __host__ __device__ __forceinline__ Ptr2dNHWC(NewAPITag, int rows_, int cols_, int ch_, int rowPitchBytes_,
                                                  T *data_)
        : batches(1)
        , rows(rows_)
        , cols(cols_)
        , ch(ch_)
        , imgPitchBytes(0)
        , rowPitchBytes(rowPitchBytes_)
        , data(data_)
    {
    }

    __host__ __forceinline__ Ptr2dNHWC(const IImageDataDevicePitch &inData)
        : batches(1)
        , rows(inData.size().h)
        , cols(inData.size().w)
        , ch(inData.format().numChannels())
        , imgPitchBytes(0)
    {
        if (inData.format().numPlanes() != 1)
        {
            throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Image must have only one plane");
        }

        const ImagePlanePitch &plane = inData.plane(0);

        rowPitchBytes = inData.plane(0).pitchBytes;
        data          = reinterpret_cast<T *>(inData.plane(0).buffer);
    }

    __host__ __forceinline__ Ptr2dNHWC(const ITensorDataPitchDevice &tensor)
    {
        DimsNCHW dims = tensor.dims();
        batches       = dims.n;
        rows          = dims.h;
        cols          = dims.w;
        ch            = dims.c;

        imgPitchBytes = tensor.pitchBytes(0);
        rowPitchBytes = tensor.pitchBytes(1);
        data          = reinterpret_cast<T *>(tensor.mem());
    }

    // ptr for uchar1/3/4, ushort1/3/4, float1/3/4, typename T -> uchar3 etc.
    // each fetch operation get a x-channel elements
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x)
    {
        //return (T *)(data + b * rows * cols + y * cols + x);
        return (T *)(reinterpret_cast<std::byte *>(data) + b * imgPitchBytes + y * rowPitchBytes + x * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x) const
    {
        //return (const T *)(data + b * rows * cols + y * cols + x);
        return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgPitchBytes + y * rowPitchBytes
                           + x * sizeof(T));
    }

    // ptr for uchar, ushort, float, typename T -> uchar etc.
    // each fetch operation get a single channel element
    __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c)
    {
        //return (T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
        return (T *)(reinterpret_cast<std::byte *>(data) + b * imgPitchBytes + y * rowPitchBytes
                     + (x * ch + c) * sizeof(T));
    }

    const __host__ __device__ __forceinline__ T *ptr(int b, int y, int x, int c) const
    {
        //return (const T *)(data + b * rows * cols * ch + y * cols * ch + x * ch + c);
        return (const T *)(reinterpret_cast<const std::byte *>(data) + b * imgPitchBytes + y * rowPitchBytes
                           + (x * ch + c) * sizeof(T));
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
    int imgPitchBytes;
    int rowPitchBytes;
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

} // namespace nv::cv::legacy::cuda_op
#endif
