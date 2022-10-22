/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "Tensor.hpp"

#include "IAllocator.hpp"
#include "Requirements.hpp"
#include "TensorData.hpp"
#include "TensorLayout.hpp"
#include "TensorShape.hpp"

#include <cuda_runtime.h>
#include <fmt/DataLayout.hpp>
#include <fmt/PixelType.hpp>
#include <util/Assert.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nv::cv::priv {

// Tensor implementation -------------------------------------------

NVCVTensorRequirements Tensor::CalcRequirements(int32_t numImages, Size2D imgSize, ImageFormat fmt)
{
    // Check if format is compatible with tensor representation
    if (fmt.memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED,
                        "Tensor image batch of block-linear format images is not currently supported.");
    }

    if (fmt.css() != NVCV_CSS_444)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED)
            << "Batch image format must not have subsampled planes, but it is: " << fmt;
    }

    if (fmt.numPlanes() != 1 && fmt.numPlanes() != fmt.numChannels())
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format cannot be semi-planar, but it is: " << fmt;
    }

    for (int p = 1; p < fmt.numPlanes(); ++p)
    {
        if (fmt.planePacking(p) != fmt.planePacking(0))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Format's planes must all have the same packing, but they don't: " << fmt;
        }
    }

    // Calculate the shape based on image parameters
    NVCVTensorLayout layout = GetTensorLayoutFor(fmt, numImages);

    int64_t shapeNCHW[4] = {numImages, fmt.numChannels(), imgSize.h, imgSize.w};

    int64_t shape[NVCV_TENSOR_MAX_NDIM];
    PermuteShape(NVCV_TENSOR_NCHW, shapeNCHW, layout, shape);

    // Calculate the element type. It's the pixel type of the
    // first channel. It assumes that all channels have same packing.
    NVCVPackingParams params = GetPackingParams(fmt.planePacking(0));
    params.swizzle           = NVCV_SWIZZLE_X000;
    std::fill(params.bits + 1, params.bits + sizeof(params.bits) / sizeof(params.bits[0]), 0);
    std::optional<NVCVPacking> chPacking = MakeNVCVPacking(params);
    if (!chPacking)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format can't be represented in a tensor: " << fmt;
    }

    PixelType dtype{fmt.dataType(), *chPacking};

    return CalcRequirements(layout.ndim, shape, dtype, layout);
}

NVCVTensorRequirements Tensor::CalcRequirements(int32_t ndim, const int64_t *shape, const PixelType &dtype,
                                                NVCVTensorLayout layout)
{
    NVCVTensorRequirements reqs;

    reqs.layout = layout;
    reqs.dtype  = dtype.value();

    if (layout.ndim > 0 && ndim != layout.ndim)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Number of shape dimensions " << ndim << " must be equal to layout dimensions " << layout.ndim;
    }

    std::copy_n(shape, ndim, reqs.shape);
    reqs.ndim = ndim;

    reqs.mem = {};

    int dev;
    NVCV_CHECK_THROW(cudaGetDevice(&dev));

    // Calculate row pitch alignment
    int rowPitchAlign;
    {
        // it usually returns 32 bytes
        NVCV_CHECK_THROW(cudaDeviceGetAttribute(&rowPitchAlign, cudaDevAttrTexturePitchAlignment, dev));

        // Makes sure it's aligned to the pixel stride
        rowPitchAlign = std::lcm(rowPitchAlign, dtype.strideBytes());
        rowPitchAlign = util::RoundUpNextPowerOfTwo(rowPitchAlign);
    }

    // Calculate base address alignment
    {
        int addrAlign;
        // it usually returns 512 bytes
        NVCV_CHECK_THROW(cudaDeviceGetAttribute(&addrAlign, cudaDevAttrTextureAlignment, dev));
        reqs.alignBytes = std::lcm(addrAlign, rowPitchAlign);
        reqs.alignBytes = util::RoundUpNextPowerOfTwo(reqs.alignBytes);

        if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                            "Alignment requirement of %d is larger than the maximum allowed %ld", reqs.alignBytes,
                            NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
        }
    }

    int firstPacked = reqs.layout == NVCV_TENSOR_NHWC ? ndim - 2 : ndim - 1;

    reqs.pitchBytes[ndim - 1] = dtype.strideBytes();
    for (int d = ndim - 2; d >= 0; --d)
    {
        if (d == firstPacked - 1)
        {
            reqs.pitchBytes[d] = util::RoundUpPowerOfTwo(reqs.shape[d + 1] * reqs.pitchBytes[d + 1], rowPitchAlign);
        }
        else
        {
            reqs.pitchBytes[d] = reqs.pitchBytes[d + 1] * reqs.shape[d + 1];
        }
    }

    AddBuffer(reqs.mem.deviceMem, reqs.pitchBytes[0] * reqs.shape[0], reqs.alignBytes);

    return reqs;
}

Tensor::Tensor(NVCVTensorRequirements reqs, IAllocator &alloc)
    : m_alloc{alloc}
    , m_reqs{std::move(reqs)}
{
    // Assuming reqs are already validated during its creation

    int64_t bufSize = CalcTotalSizeBytes(m_reqs.mem.deviceMem);
    m_buffer        = m_alloc.allocDeviceMem(bufSize, m_reqs.alignBytes);
    NVCV_ASSERT(m_buffer != nullptr);
}

Tensor::~Tensor()
{
    m_alloc.freeDeviceMem(m_buffer, CalcTotalSizeBytes(m_reqs.mem.deviceMem), m_reqs.alignBytes);
}

Version Tensor::doGetVersion() const
{
    return CURRENT_VERSION;
}

int32_t Tensor::ndim() const
{
    return m_reqs.ndim;
}

const int64_t *Tensor::shape() const
{
    return m_reqs.shape;
}

const NVCVTensorLayout &Tensor::layout() const
{
    return m_reqs.layout;
}

DimsNCHW Tensor::dims() const
{
    return ToNCHW(this->shape(), this->layout());
}

PixelType Tensor::dtype() const
{
    return PixelType{m_reqs.dtype};
}

IAllocator &Tensor::alloc() const
{
    return m_alloc;
}

void Tensor::exportData(NVCVTensorData &data) const
{
    data.bufferType = NVCV_TENSOR_BUFFER_PITCH_DEVICE;

    data.dtype  = m_reqs.dtype;
    data.layout = m_reqs.layout;
    data.ndim   = m_reqs.ndim;

    memcpy(data.shape, m_reqs.shape, sizeof(data.shape));

    NVCVTensorBufferPitch &buf = data.buffer.pitch;
    {
        static_assert(sizeof(buf.pitchBytes) == sizeof(m_reqs.pitchBytes));
        static_assert(
            std::is_same_v<std::decay_t<decltype(buf.pitchBytes[0])>, std::decay_t<decltype(m_reqs.pitchBytes[0])>>);
        memcpy(buf.pitchBytes, m_reqs.pitchBytes, sizeof(buf.pitchBytes));

        buf.data = m_buffer;
    }
}

} // namespace nv::cv::priv
