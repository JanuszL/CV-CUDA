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

    int32_t shape[4];
    shape[0] = numImages;
    switch (layout)
    {
    case NVCV_TENSOR_NCHW:
        shape[1] = fmt.numChannels();
        shape[2] = imgSize.h;
        shape[3] = imgSize.w;
        break;

    case NVCV_TENSOR_NHWC:
        shape[1] = imgSize.h;
        shape[2] = imgSize.w;
        shape[3] = fmt.numChannels();
        break;
    }

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

    return CalcRequirements(shape, layout, dtype);
}

NVCVTensorRequirements Tensor::CalcRequirements(const int32_t *shape, NVCVTensorLayout layout, const PixelType &dtype)
{
    NVCVTensorRequirements reqs;

    reqs.layout = layout;
    reqs.dtype  = dtype.value();

    std::copy(shape, shape + GetNumDim(layout), reqs.shape);

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

    switch (reqs.layout)
    {
    case NVCV_TENSOR_NCHW:
        // chPitch
        reqs.pitchBytes[3] = dtype.strideBytes();
        // rowPitch = width * chPitch
        reqs.pitchBytes[2] = util::RoundUpPowerOfTwo(reqs.shape[3] * reqs.pitchBytes[3], rowPitchAlign);
        // planePitch = rowPitch*height
        reqs.pitchBytes[1] = reqs.pitchBytes[2] * reqs.shape[2];
        // imgPitch = planePitch*numPlanes
        reqs.pitchBytes[0] = reqs.pitchBytes[1] * reqs.shape[1];
        break;

    case NVCV_TENSOR_NHWC:
        // chPitch
        reqs.pitchBytes[3] = dtype.strideBytes();
        // pixPitch = chPitch*numChannels
        reqs.pitchBytes[2] = reqs.pitchBytes[3] * reqs.shape[3];
        // rowPitch = pixPitch * width @ pitchAlign
        reqs.pitchBytes[1] = util::RoundUpPowerOfTwo(reqs.shape[2] * reqs.pitchBytes[2], rowPitchAlign);
        // imgPitch = rowPitch*height
        reqs.pitchBytes[0] = reqs.pitchBytes[1] * reqs.shape[1];
        break;
    }

    // imgPitch * numImages
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

const Shape &Tensor::shape() const
{
    static_assert(sizeof(Shape) == sizeof(m_reqs.shape));
    static_assert(std::is_same_v<Shape::value_type, std::decay_t<decltype(m_reqs.shape[0])>>);
    return *reinterpret_cast<const Shape *>(m_reqs.shape);
}

NVCVTensorLayout Tensor::layout() const
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

    NVCVTensorBufferPitch &buf = data.buffer.pitch;
    {
        buf.dtype  = m_reqs.dtype;
        buf.layout = m_reqs.layout;

        memcpy(buf.shape, m_reqs.shape, sizeof(buf.shape));

        static_assert(sizeof(buf.pitchBytes) == sizeof(m_reqs.pitchBytes));
        static_assert(
            std::is_same_v<std::decay_t<decltype(buf.pitchBytes[0])>, std::decay_t<decltype(m_reqs.pitchBytes[0])>>);
        memcpy(buf.pitchBytes, m_reqs.pitchBytes, sizeof(buf.pitchBytes));

        buf.mem = m_buffer;
    }
}

} // namespace nv::cv::priv
