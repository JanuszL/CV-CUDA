/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
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

#include "../CvCudaLegacy.h"
#include "../CvCudaLegacyHelpers.hpp"

#include "../CvCudaUtils.cuh"

using namespace nv::cv::legacy::cuda_op;
using namespace nv::cv::legacy::helpers;

namespace nvcv = nv::cv;

#define BLOCK 32

#define USE_OCV_CPU_ALIGN_VERSION

template<typename T>
__global__ void resize_linear_ocv_align(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const float scale_x,
                                        const float scale_y)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = src.rows, width = src.cols, out_height = dst.rows, out_width = dst.cols;

    if (dst_x < out_width && dst_y < out_height)
    {
        using work_type = nv::cv::cuda::ConvertBaseTypeTo<float, T>;
        work_type out   = nv::cv::cuda::SetAll<work_type>(0);

        float fy = (float)((dst_y + 0.5) * scale_y - 0.5);
        int   sy = __float2int_rd(fy);
        fy -= sy;
        sy = min(sy, height - 2);
        sy = max(0, sy);

        float cbufy[2];
        cbufy[0] = 1.f - fy;
        cbufy[1] = fy;

        float fx = (float)((dst_x + 0.5) * scale_x - 0.5);
        int   sx = __float2int_rd(fx);
        fx -= sx;

        if (sx < 0)
        {
            fx = 0, sx = 0;
        }
        if (sx >= width - 1)
        {
            fx = 0, sx = width - 2;
        }

        float cbufx[2];
        cbufx[0] = 1.f - fx;
        cbufx[1] = fx;

        *dst.ptr(batch_idx, dst_y, dst_x) = nv::cv::cuda::SaturateCast<nv::cv::cuda::BaseType<T>>(
            (*src.ptr(batch_idx, sy, sx) * cbufx[0] * cbufy[0] + *src.ptr(batch_idx, sy + 1, sx) * cbufx[0] * cbufy[1]
             + *src.ptr(batch_idx, sy, sx + 1) * cbufx[1] * cbufy[0]
             + *src.ptr(batch_idx, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
    }
}

template<typename T>
__global__ void resize_linear_v2(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const float scale_x, const float scale_y)
{
    const int dst_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    int       height = src.rows, width = src.cols, out_height = dst.rows, out_width = dst.cols;

    if (dst_x < out_width && dst_y < out_height)
    {
        const float src_x = dst_x * scale_x;
        const float src_y = dst_y * scale_y;

        using work_type = nv::cv::cuda::ConvertBaseTypeTo<float, T>;
        work_type out   = nv::cv::cuda::SetAll<work_type>(0);

        const int x1      = __float2int_rd(src_x);
        const int y1      = __float2int_rd(src_y);
        const int x2      = x1 + 1;
        const int y2      = y1 + 1;
        const int x2_read = min(x2, width - 1);
        const int y2_read = min(y2, height - 1);

        T src_reg = *src.ptr(batch_idx, y1, x1);
        out       = out + src_reg * ((x2 - src_x) * (y2 - src_y));

        src_reg = *src.ptr(batch_idx, y1, x2_read);
        out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

        src_reg = *src.ptr(batch_idx, y2_read, x1);
        out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

        src_reg = *src.ptr(batch_idx, y2_read, x2_read);
        out     = out + src_reg * ((src_x - x1) * (src_y - y1));

        *dst.ptr(batch_idx, dst_y, dst_x) = nv::cv::cuda::SaturateCast<nv::cv::cuda::BaseType<T>>(out);
    }
}

template<typename T>
__global__ void resize_nearest_ocv_align(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const float scale_x,
                                         const float scale_y)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    if (dst_x < out_width && dst_y < out_height)
    {
        int sx = __float2int_rd(dst_x * scale_x);
        sx     = min(sx, src.cols - 1);

        int sy                            = __float2int_rd(dst_y * scale_y);
        sy                                = min(sy, src.rows - 1);
        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, sy, sx);
    }
}

template<typename T>
__global__ void resize_nearest_v2(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const float scale_x, const float scale_y)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    if (dst_x < out_width && dst_y < out_height)
    {
        const float src_x = dst_x * scale_x;
        const float src_y = dst_y * scale_y;

        const int x1 = __float2int_rz(src_x);
        const int y1 = __float2int_rz(src_y);

        *dst.ptr(batch_idx, dst_y, dst_x) = *src.ptr(batch_idx, y1, x1);
    }
}

template<typename T>
__global__ void resize_cubic_ocv_align(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, const float scale_x,
                                       const float scale_y)
{
    const int x          = blockIdx.x * blockDim.x + threadIdx.x;
    const int y          = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;
    if (x >= out_width || y >= out_height)
        return;

    int iscale_x = nv::cv::cuda::SaturateCast<int>(scale_x);
    int iscale_y = nv::cv::cuda::SaturateCast<int>(scale_y);

    float fy = (float)((y + 0.5) * scale_y - 0.5);
    int   sy = __float2int_rd(fy);
    fy -= sy;
    sy = min(sy, src.rows - 3);
    sy = max(1, sy);

    const float A = -0.75f;

    float coeffsY[4];
    coeffsY[0] = ((A * (fy + 1) - 5 * A) * (fy + 1) + 8 * A) * (fy + 1) - 4 * A;
    coeffsY[1] = ((A + 2) * fy - (A + 3)) * fy * fy + 1;
    coeffsY[2] = ((A + 2) * (1 - fy) - (A + 3)) * (1 - fy) * (1 - fy) + 1;
    coeffsY[3] = 1.f - coeffsY[0] - coeffsY[1] - coeffsY[2];

    float fx = (float)((x + 0.5) * scale_x - 0.5);
    int   sx = __float2int_rd(fx);
    fx -= sx;

    if (sx < 1)
    {
        fx = 0, sx = 1;
    }
    if (sx >= src.cols - 3)
    {
        fx = 0, sx = src.cols - 3;
    }

    float coeffsX[4];
    coeffsX[0] = ((A * (fx + 1) - 5 * A) * (fx + 1) + 8 * A) * (fx + 1) - 4 * A;
    coeffsX[1] = ((A + 2) * fx - (A + 3)) * fx * fx + 1;
    coeffsX[2] = ((A + 2) * (1 - fx) - (A + 3)) * (1 - fx) * (1 - fx) + 1;
    coeffsX[3] = 1.f - coeffsX[0] - coeffsX[1] - coeffsX[2];

    if (sx < 1)
    {
        sx = 1;
    }
    if (sx > src.cols - 3)
    {
        sx = src.cols - 3;
    }
    if (sy < 1)
    {
        sy = 1;
    }
    if (sy > src.rows - 3)
    {
        sy = src.rows - 3;
    }

    *dst.ptr(batch_idx, y, x) = nv::cv::cuda::SaturateCast<nv::cv::cuda::BaseType<T>>(
        nv::cv::cuda::abs(*src.ptr(batch_idx, sy - 1, sx - 1) * coeffsX[0] * coeffsY[0]
                          + *src.ptr(batch_idx, sy, sx - 1) * coeffsX[0] * coeffsY[1]
                          + *src.ptr(batch_idx, sy + 1, sx - 1) * coeffsX[0] * coeffsY[2]
                          + *src.ptr(batch_idx, sy + 2, sx - 1) * coeffsX[0] * coeffsY[3]
                          + *src.ptr(batch_idx, sy - 1, sx) * coeffsX[1] * coeffsY[0]
                          + *src.ptr(batch_idx, sy, sx) * coeffsX[1] * coeffsY[1]
                          + *src.ptr(batch_idx, sy + 1, sx) * coeffsX[1] * coeffsY[2]
                          + *src.ptr(batch_idx, sy + 2, sx) * coeffsX[1] * coeffsY[3]
                          + *src.ptr(batch_idx, sy - 1, sx + 1) * coeffsX[2] * coeffsY[0]
                          + *src.ptr(batch_idx, sy, sx + 1) * coeffsX[2] * coeffsY[1]
                          + *src.ptr(batch_idx, sy + 1, sx + 1) * coeffsX[2] * coeffsY[2]
                          + *src.ptr(batch_idx, sy + 2, sx + 1) * coeffsX[2] * coeffsY[3]
                          + *src.ptr(batch_idx, sy - 1, sx + 2) * coeffsX[3] * coeffsY[0]
                          + *src.ptr(batch_idx, sy, sx + 2) * coeffsX[3] * coeffsY[1]
                          + *src.ptr(batch_idx, sy + 1, sx + 2) * coeffsX[3] * coeffsY[2]
                          + *src.ptr(batch_idx, sy + 2, sx + 2) * coeffsX[3] * coeffsY[3]));
}

template<typename T>
__global__ void resize_cubic_v2(CubicFilter<BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>>> filteredSrc, Ptr2dNHWC<T> dst,
                                const float scale_x, const float scale_y)
{
    const int dst_x      = blockIdx.x * blockDim.x + threadIdx.x;
    const int dst_y      = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    if (dst_x < out_width && dst_y < out_height)
    {
        const float src_x = dst_x * scale_x;
        const float src_y = dst_y * scale_y;

        *dst.ptr(batch_idx, dst_y, dst_x) = filteredSrc(batch_idx, src_y, src_x);
    }
}

template<typename T, typename IntegerAreaFilter, typename AreaFilter>
__global__ void resize_area_ocv_align(const Ptr2dNHWC<T> src, const IntegerAreaFilter integer_filter,
                                      const AreaFilter area_filter, Ptr2dNHWC<T> dst, const float scale_x,
                                      const float scale_y)
{
    const int x          = blockDim.x * blockIdx.x + threadIdx.x;
    const int y          = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    if (x >= out_width || y >= out_height)
        return;

    double inv_scale_x = 1. / scale_x;
    double inv_scale_y = 1. / scale_y;
    int    iscale_x    = nv::cv::cuda::SaturateCast<int>(scale_x);
    int    iscale_y    = nv::cv::cuda::SaturateCast<int>(scale_y);
    bool   is_area_fast
        = nv::cv::cuda::abs(scale_x - iscale_x) < DBL_EPSILON && nv::cv::cuda::abs(scale_y - iscale_y) < DBL_EPSILON;

    if (scale_x >= 1.0f && scale_y >= 1.0f) // zoom out
    {
        if (is_area_fast) // integer multiples
        {
            *dst.ptr(batch_idx, y, x) = integer_filter(batch_idx, y, x);
            return;
        }

        *dst.ptr(batch_idx, y, x) = area_filter(batch_idx, y, x);
        return;
    }

    // zoom in, it is emulated using some variant of bilinear interpolation
    int   sy = __float2int_rd(y * scale_y);
    float fy = (float)((y + 1) - (sy + 1) * inv_scale_y);
    fy       = fy <= 0 ? 0.f : fy - __float2int_rd(fy);

    float cbufy[2];
    cbufy[0] = 1.f - fy;
    cbufy[1] = fy;

    int   sx = __float2int_rd(x * scale_x);
    float fx = (float)((x + 1) - (sx + 1) * inv_scale_x);
    fx       = fx < 0 ? 0.f : fx - __float2int_rd(fx);

    if (sx < 0)
    {
        fx = 0, sx = 0;
    }

    if (sx >= src.cols - 1)
    {
        fx = 0, sx = src.cols - 2;
    }
    if (sy >= src.rows - 1)
    {
        sy = src.rows - 2;
    }

    float cbufx[2];
    cbufx[0] = 1.f - fx;
    cbufx[1] = fx;

    *dst.ptr(batch_idx, y, x) = nv::cv::cuda::SaturateCast<nv::cv::cuda::BaseType<T>>(
        (*src.ptr(batch_idx, sy, sx) * cbufx[0] * cbufy[0] + *src.ptr(batch_idx, sy + 1, sx) * cbufx[0] * cbufy[1]
         + *src.ptr(batch_idx, sy, sx + 1) * cbufx[1] * cbufy[0]
         + *src.ptr(batch_idx, sy + 1, sx + 1) * cbufx[1] * cbufy[1]));
}

template<class Filter, typename T>
__global__ void resize_area_v2(const Filter src, Ptr2dNHWC<T> dst)
{
    const int x          = blockDim.x * blockIdx.x + threadIdx.x;
    const int y          = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx  = get_batch_idx();
    int       out_height = dst.rows, out_width = dst.cols;

    if (x < out_width && y < out_height)
    {
        *dst.ptr(batch_idx, y, x) = src(batch_idx, y, x);
    }
}

template<typename T>
void resize(const nvcv::TensorDataAccessPitchImagePlanar &inData, const nvcv::TensorDataAccessPitchImagePlanar &outData,
            NVCVInterpolationType interpolation, cudaStream_t stream)
{
    const int batch_size = inData.numSamples();
    const int in_width   = inData.numCols();
    const int in_height  = inData.numRows();
    const int out_width  = outData.numCols();
    const int out_height = outData.numRows();

    const dim3 blockSize(BLOCK, BLOCK / 4, 1);
    const dim3 gridSize(divUp(out_width, blockSize.x), divUp(out_height, blockSize.y), batch_size);

    Ptr2dNHWC<T> src_ptr(inData);  //batch_size, height, width, channels, (T *)d_in);
    Ptr2dNHWC<T> dst_ptr(outData); //batch_size, out_height, out_width, channels, (T *)d_out);

    float scale_x = static_cast<float>(in_width) / out_width;
    float scale_y = static_cast<float>(in_height) / out_height;
// change to linear if area interpolation upscaling
#ifndef USE_OCV_CPU_ALIGN_VERSION
    if (interpolation == NVCV_INTERP_AREA && (scale_x <= 1.f || scale_y <= 1.f))
    {
        interpolation = NVCV_INTERP_LINEAR;
    }
#endif

    if (interpolation == NVCV_INTERP_LINEAR)
    {
#ifdef USE_OCV_CPU_ALIGN_VERSION
        resize_linear_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, scale_x, scale_y);
#else
        resize_linear_v2<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, scale_x, scale_y);
#endif
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_NEAREST)
    {
#ifdef USE_OCV_CPU_ALIGN_VERSION
        resize_nearest_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, scale_x, scale_y);
#else
        resize_nearest_v2<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, scale_x, scale_y);
#endif
        checkKernelErrors();
    }
    else if (interpolation == NVCV_INTERP_CUBIC)
    {
#ifdef USE_OCV_CPU_ALIGN_VERSION
        resize_cubic_ocv_align<T><<<gridSize, blockSize, 0, stream>>>(src_ptr, dst_ptr, scale_x, scale_y);
        checkKernelErrors();
#else
        BrdReplicate<T>                                          brd(src_ptr.rows, src_ptr.cols);
        BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>>              brdSrc(src_ptr, brd);
        CubicFilter<BorderReader<Ptr2dNHWC<T>, BrdReplicate<T>>> filteredSrc(brdSrc);
        resize_cubic_v2<T><<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr, scale_x, scale_y);
        checkKernelErrors();
#endif
    }
    else if (interpolation == NVCV_INTERP_AREA)
    {
#ifdef USE_OCV_CPU_ALIGN_VERSION
        BrdConstant<T>                                                brd(src_ptr.rows, src_ptr.cols);
        BorderReader<Ptr2dNHWC<T>, BrdConstant<T>>                    brdSrc(src_ptr, brd);
        IntegerAreaFilter<BorderReader<Ptr2dNHWC<T>, BrdConstant<T>>> integer_filter(brdSrc, scale_x, scale_y);
        AreaFilter<BorderReader<Ptr2dNHWC<T>, BrdConstant<T>>>        area_filter(brdSrc, scale_x, scale_y);
        resize_area_ocv_align<T>
            <<<gridSize, blockSize, 0, stream>>>(src_ptr, integer_filter, area_filter, dst_ptr, scale_x, scale_y);
        checkKernelErrors();
#else
        const int iscale_x = (int)round(scale_x);
        const int iscale_y = (int)round(scale_y);
        if (std::abs(scale_x - iscale_x) < FLT_MIN && std::abs(scale_y - iscale_y) < FLT_MIN)
        {
            BrdConstant<T>                                                brd(src_ptr.rows, src_ptr.cols);
            BorderReader<Ptr2dNHWC<T>, BrdConstant<T>>                    brdSrc(src_ptr, brd);
            IntegerAreaFilter<BorderReader<Ptr2dNHWC<T>, BrdConstant<T>>> filteredSrc(brdSrc, scale_x, scale_y);

            resize_area_v2<<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr);
            checkKernelErrors();
        }
        else
        {
            BrdConstant<T>                                         brd(src_ptr.rows, src_ptr.cols);
            BorderReader<Ptr2dNHWC<T>, BrdConstant<T>>             brdSrc(src_ptr, brd);
            AreaFilter<BorderReader<Ptr2dNHWC<T>, BrdConstant<T>>> filteredSrc(brdSrc, scale_x, scale_y);

            resize_area_v2<<<gridSize, blockSize, 0, stream>>>(filteredSrc, dst_ptr);
            checkKernelErrors();
        }
#endif
    }

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nv::cv::legacy::cuda_op {

size_t Resize::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode Resize::infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                        const NVCVInterpolationType interpolation, cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessPitchImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessPitchImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    int channels = input_shape.C;

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

    typedef void (*func_t)(const nvcv::TensorDataAccessPitchImagePlanar &inData,
                           const nvcv::TensorDataAccessPitchImagePlanar &outData,
                           const NVCVInterpolationType interpolation, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      resize<uchar>,  0 /*resize<uchar2>*/,      resize<uchar3>,      resize<uchar4>},
        {0 /*resize<schar>*/,   0 /*resize<char2>*/, 0 /*resize<char3>*/, 0 /*resize<char4>*/},
        {     resize<ushort>, 0 /*resize<ushort2>*/,     resize<ushort3>,     resize<ushort4>},
        {      resize<short>,  0 /*resize<short2>*/,      resize<short3>,      resize<short4>},
        {  0 /*resize<int>*/,    0 /*resize<int2>*/,  0 /*resize<int3>*/,  0 /*resize<int4>*/},
        {      resize<float>,  0 /*resize<float2>*/,      resize<float3>,      resize<float4>}
    };

    if (interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
        || interpolation == NVCV_INTERP_CUBIC || interpolation == NVCV_INTERP_AREA)
    {
        const func_t func = funcs[data_type][channels - 1];
        NVCV_ASSERT(func != 0);

        func(*inAccess, *outAccess, interpolation, stream);
    }
    else
    {
        LOG_ERROR("Invalid interpolation " << interpolation);
        return ErrorCode::INVALID_PARAMETER;
    }
    return SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
