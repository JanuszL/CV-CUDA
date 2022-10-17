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

#include <cuda_runtime.h>
#include <util/Assert.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nv::cv::priv {

// Tensor implementation -------------------------------------------

NVCVTensorRequirements Tensor::CalcRequirements(int32_t numImages, Size2D imgSize, ImageFormat fmt)
{
    DimsNCHW dims;

    dims.n = numImages;
    dims.c = fmt.numChannels();
    dims.h = imgSize.h;
    dims.w = imgSize.w;

    return CalcRequirements(dims, fmt);
}

NVCVTensorRequirements Tensor::CalcRequirements(const DimsNCHW &dims, ImageFormat fmt)
{
    ValidateImageFormatForTensor(fmt);

    if (dims.c != fmt.numChannels())
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Number of channels " << dims.c << " doesn't match number of channels in format" << fmt;
    }

    NVCVTensorRequirements reqs;

    reqs.shape[0] = dims.n;

    // planar?
    if (fmt.numPlanes() == fmt.numChannels())
    {
        reqs.layout   = NVCV_TENSOR_NCHW;
        reqs.shape[1] = dims.c;
        reqs.shape[2] = dims.h;
        reqs.shape[3] = dims.w;
    }
    else
    {
        NVCV_ASSERT(fmt.numPlanes() == 1);
        reqs.layout   = NVCV_TENSOR_NHWC;
        reqs.shape[1] = dims.h;
        reqs.shape[2] = dims.w;
        reqs.shape[3] = dims.c;
    }

    reqs.format = fmt.value();
    reqs.mem    = {};

    int dev;
    NVCV_CHECK_THROW(cudaGetDevice(&dev));

    // Calculate row pitch alignment
    int rowPitchAlign;
    {
        // it usually returns 32 bytes
        NVCV_CHECK_THROW(cudaDeviceGetAttribute(&rowPitchAlign, cudaDevAttrTexturePitchAlignment, dev));

        // Makes sure it's aligned to the pixel stride
        rowPitchAlign = std::lcm(rowPitchAlign, fmt.planePixelStrideBytes(0));
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
        reqs.pitchBytes[3] = fmt.planePixelStrideBytes(0);
        // rowPitch = width * chPitch
        reqs.pitchBytes[2] = util::RoundUpPowerOfTwo(reqs.shape[3] * reqs.pitchBytes[3], rowPitchAlign);
        // planePitch = rowPitch*height
        reqs.pitchBytes[1] = reqs.pitchBytes[2] * reqs.shape[2];
        // imgPitch = planePitch*numPlanes
        reqs.pitchBytes[0] = reqs.pitchBytes[1] * reqs.shape[1];
        break;

    case NVCV_TENSOR_NHWC:
        // pixPitch
        reqs.pitchBytes[2] = fmt.planePixelStrideBytes(0);
        // chPitch = pixPitch / num_ch
        reqs.pitchBytes[3] = reqs.pitchBytes[2] / fmt.numChannels();
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
    ImageFormat fmt{m_reqs.format};

    // Assuming the format is already validated during requirements creation.

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

ImageFormat Tensor::format() const
{
    return ImageFormat{m_reqs.format};
}

IAllocator &Tensor::alloc() const
{
    return m_alloc;
}

void Tensor::exportData(NVCVTensorData &data) const
{
    ImageFormat fmt{m_reqs.format};

    NVCV_ASSERT(fmt.memLayout() == NVCV_MEM_LAYOUT_PL);

    data.format     = m_reqs.format;
    data.bufferType = NVCV_TENSOR_BUFFER_PITCH_DEVICE;

    NVCVTensorBufferPitch &buf = data.buffer.pitch;
    {
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
