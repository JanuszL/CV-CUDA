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

namespace nv::cv::legacy::cuda_op {

template<class SrcWrapper, class DstWrapper, class KernelWrapper>
__global__ void filter2D(SrcWrapper src, DstWrapper dst, Size2D dstSize, KernelWrapper kernel, Size2D kernelSize,
                         int2 kernelAnchor)
{
    using T         = typename DstWrapper::ValueType;
    using BT        = cuda::BaseType<T>;
    using work_type = cuda::ConvertBaseTypeTo<float, T>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dstSize.w || y >= dstSize.h)
        return;

    int  kInd = 0;
    int3 coord{x, y, batch_idx};

    for (int i = 0; i < kernelSize.h; ++i)
    {
        coord.y = y - kernelAnchor.y + i;

        for (int j = 0; j < kernelSize.w; ++j)
        {
            coord.x = x - kernelAnchor.x + j;

            res = res + src[coord] * kernel[kInd++];
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<BT>(res);
}

template<typename T, NVCVBorderType B, class KernelWrapper>
void Filter2DCaller(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, KernelWrapper kernel,
                    Size2D kernelSize, int2 kernelAnchor, float borderValue, cudaStream_t stream)
{
    auto outAccess = TensorDataAccessPitchImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    Size2D dstSize{outAccess->numCols(), outAccess->numRows()};

    cuda::BorderWrapNHW<const T, B> src(inData, cuda::SetAll<T>(borderValue));
    cuda::Tensor3DWrap<T>           dst(outData);

    dim3 block(16, 16);
    dim3 grid(divUp(dstSize.w, block.x), divUp(dstSize.h, block.y), outAccess->numSamples());

    filter2D<<<grid, block, 0, stream>>>(src, dst, dstSize, kernel, kernelSize, kernelAnchor);

    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename T, class KernelWrapper>
void Filter2D(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, KernelWrapper kernel,
              Size2D kernelSize, int2 kernelAnchor, NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    switch (borderMode)
    {
#define NVCV_FILTER_CASE(BORDERTYPE)                                                                           \
    case BORDERTYPE:                                                                                           \
        Filter2DCaller<T, BORDERTYPE>(inData, outData, kernel, kernelSize, kernelAnchor, borderValue, stream); \
        break

        NVCV_FILTER_CASE(NVCV_BORDER_CONSTANT);
        NVCV_FILTER_CASE(NVCV_BORDER_REPLICATE);
        NVCV_FILTER_CASE(NVCV_BORDER_REFLECT);
        NVCV_FILTER_CASE(NVCV_BORDER_WRAP);
        NVCV_FILTER_CASE(NVCV_BORDER_REFLECT101);

#undef NVCV_FILTER_CASE
    default:
        break;
    }
}

// Laplacian -------------------------------------------------------------------

// @brief Laplacian 3x3 kernels for ksize == 1 and ksize == 3

// clang-format off
constexpr Size2D kLaplacianKernelSize{3, 3};

constexpr cuda::math::Vector<float, 9> kLaplacianKernel1{
    {0.0f,  1.0f, 0.0f,
     1.0f, -4.0f, 1.0f,
     0.0f,  1.0f, 0.0f}
};
constexpr cuda::math::Vector<float, 9> kLaplacianKernel3{
    {2.0f,  0.0f, 2.0f,
     0.0f, -8.0f, 0.0f,
     2.0f,  0.0f, 2.0f}
};

// clang-format on

size_t Laplacian::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode Laplacian::infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, int ksize,
                           float scale, NVCVBorderType borderMode, cudaStream_t stream)
{
    if (!(ksize == 1 || ksize == 3))
    {
        LOG_ERROR("Invalid ksize " << ksize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Invalid DataType between input (" << inData.dtype() << ") and output (" << outData.dtype() << ")");
        return ErrorCode::INVALID_DATA_TYPE;
    }

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

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    auto inAccess = TensorDataAccessPitchImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = input_shape.C;

    int2 kernelAnchor{-1, -1};
    normalizeAnchor(kernelAnchor, kLaplacianKernelSize);
    float borderValue = .0f;

    typedef void (*filter2D_t)(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                               cuda::math::Vector<float, 9> kernel, Size2D kernelSize, int2 kernelAnchor,
                               NVCVBorderType borderMode, float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { Filter2D<uchar>, 0,  Filter2D<uchar3>,  Filter2D<uchar4>},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        {               0, 0,                 0,                 0},
        {               0, 0,                 0,                 0},
        { Filter2D<float>, 0,  Filter2D<float3>,  Filter2D<float4>},
    };

    cuda::math::Vector<float, 9> kernel;

    if (ksize == 1)
    {
        kernel = kLaplacianKernel1;
    }
    else if (ksize == 3)
    {
        kernel = kLaplacianKernel3;
    }

    if (scale != 1)
    {
        kernel *= scale;
    }

    funcs[data_type][channels - 1](inData, outData, kernel, kLaplacianKernelSize, kernelAnchor, borderMode, borderValue,
                                   stream);

    return ErrorCode::SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
