/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Image.hpp"

#include "IAllocator.hpp"
#include "Requirements.hpp"

#include <cuda_runtime.h>
#include <fmt/DataType.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nv::cv::priv {

// Image implementation -------------------------------------------

NVCVImageRequirements Image::CalcRequirements(Size2D size, ImageFormat fmt, int32_t userBaseAlign, int32_t userRowAlign)
{
    NVCVImageRequirements reqs;
    reqs.width  = size.w;
    reqs.height = size.h;
    reqs.format = fmt.value();
    reqs.mem    = {};

    int dev;
    NVCV_CHECK_THROW(cudaGetDevice(&dev));

    int rowAlign;
    if (userRowAlign == 0)
    {
        NVCV_CHECK_THROW(cudaDeviceGetAttribute(&rowAlign, cudaDevAttrTexturePitchAlignment, dev));
    }
    else
    {
        if (!util::IsPowerOfTwo(userRowAlign))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Invalid pitch alignment of " << userRowAlign << ", it must be a power-of-two";
        }
        rowAlign = userRowAlign;
    }

    // Pitch alignment must be compatible with each plane's pixel stride.
    for (int p = 0; p < fmt.numPlanes(); ++p)
    {
        int rowAlign;
        if (userRowAlign == 0)
        {
            // Safest thing we can do
            rowAlign = fmt.planePixelStrideBytes(p);
        }
        else
        {
            // Strictest thing we can do
            rowAlign = fmt.planeRowAlignment(p);
        }

        rowAlign = std::lcm(rowAlign, rowAlign);
    }

    rowAlign = util::RoundUpNextPowerOfTwo(rowAlign);

    int baseAlign;
    if (userBaseAlign == 0)
    {
        NVCV_CHECK_THROW(cudaDeviceGetAttribute(&baseAlign, cudaDevAttrTextureAlignment, dev));
    }
    else
    {
        if (!util::IsPowerOfTwo(userBaseAlign))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Invalid buffer address alignment of " << userBaseAlign << ", it must be a power-of-two";
        }
        baseAlign = userBaseAlign;
    }

    // buffer address alignment must be at least the row alignment
    baseAlign = std::lcm(baseAlign, rowAlign);

    reqs.alignBytes = baseAlign;
    if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment requirement of %d is larger than the maximum allowed %ld", reqs.alignBytes,
                        NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
    }

    // Calculate total device memory needed in blocks
    for (int p = 0; p < fmt.numPlanes(); ++p)
    {
        Size2D planeSize = fmt.planeSize(size, p);

        NVCV_ASSERT((size_t)p < sizeof(reqs.planeRowPitchBytes) / sizeof(reqs.planeRowPitchBytes[0]));

        reqs.planeRowPitchBytes[p]
            = util::RoundUpPowerOfTwo((int64_t)planeSize.w * fmt.planePixelStrideBytes(p), rowAlign);

        AddBuffer(reqs.mem.deviceMem, reqs.planeRowPitchBytes[p] * planeSize.h, baseAlign);
    }

    return reqs;
}

Image::Image(NVCVImageRequirements reqs, IAllocator &alloc)
    : m_alloc{alloc}
    , m_reqs{std::move(reqs)}
{
    if (ImageFormat{m_reqs.format}.memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED, "Image with block-linear format is not currently supported.");
    }

    int64_t bufSize = CalcTotalSizeBytes(m_reqs.mem.deviceMem);
    m_buffer        = m_alloc.allocDeviceMem(bufSize, m_reqs.alignBytes);
    NVCV_ASSERT(m_buffer != nullptr);
}

Image::~Image()
{
    m_alloc.freeDeviceMem(m_buffer, CalcTotalSizeBytes(m_reqs.mem.deviceMem), m_reqs.alignBytes);
}

NVCVTypeImage Image::type() const
{
    return NVCV_TYPE_IMAGE;
}

Size2D Image::size() const
{
    return {m_reqs.width, m_reqs.height};
}

ImageFormat Image::format() const
{
    return ImageFormat{m_reqs.format};
}

IAllocator &Image::alloc() const
{
    return m_alloc;
}

void Image::exportData(NVCVImageData &data) const
{
    ImageFormat fmt{m_reqs.format};

    NVCV_ASSERT(fmt.memLayout() == NVCV_MEM_LAYOUT_PL);

    data.format     = m_reqs.format;
    data.bufferType = NVCV_IMAGE_BUFFER_PITCH_DEVICE;

    NVCVImageBufferPitch &buf = data.buffer.pitch;

    buf.numPlanes            = fmt.numPlanes();
    int64_t planeOffsetBytes = 0;
    for (int p = 0; p < buf.numPlanes; ++p)
    {
        NVCVImagePlanePitch &plane = buf.planes[p];

        Size2D planeSize = fmt.planeSize({m_reqs.width, m_reqs.height}, p);

        plane.width      = planeSize.w;
        plane.height     = planeSize.h;
        plane.pitchBytes = m_reqs.planeRowPitchBytes[p];
        plane.buffer     = reinterpret_cast<std::byte *>(m_buffer) + planeOffsetBytes;

        planeOffsetBytes += plane.height * plane.pitchBytes;
    }

    // Due to addr alignment, the allocated buffer could be larger than what we need,
    // but it can't be smaller.
    NVCV_ASSERT(planeOffsetBytes <= CalcTotalSizeBytes(m_reqs.mem.deviceMem));
}

// ImageWrap implementation -------------------------------------------

ImageWrapData::ImageWrapData(const NVCVImageData &data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup)
    : m_cleanup(cleanup)
    , m_ctxCleanup(ctxCleanup)
{
    doValidateData(data);

    m_data = data;
}

ImageWrapData::~ImageWrapData()
{
    doCleanup();
}

void ImageWrapData::doValidateData(const NVCVImageData &data) const
{
    ImageFormat format{data.format};

    bool success = false;
    switch (data.bufferType)
    {
    case NVCV_IMAGE_BUFFER_PITCH_DEVICE:
        if (format.memLayout() != NVCV_MEM_LAYOUT_PL)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Image buffer type PITCH_DEVICE not consistent with image format " << format;
        }

        if (data.buffer.pitch.numPlanes < 1)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Number of planes must be >= 1, not " << data.buffer.pitch.numPlanes;
        }

        for (int p = 0; p < data.buffer.pitch.numPlanes; ++p)
        {
            const NVCVImagePlanePitch &plane = data.buffer.pitch.planes[p];
            if (plane.width < 1 || plane.height < 1)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                    << "Plane #" << p << " must have dimensions >= 1x1, not " << plane.width << "x" << plane.height;
            }

            if (plane.buffer == nullptr)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Plane #" << p << "'s buffer pointer must not be NULL";
            }
        }
        success = true;
        break;

    case NVCV_IMAGE_BUFFER_PITCH_HOST:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Wrapping of host memory into an image isn't currently supported";

    case NVCV_IMAGE_BUFFER_CUDA_ARRAY:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Wrapping of cudaArray into an image isn't currently supported";

    case NVCV_IMAGE_BUFFER_NONE:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid wrapping of buffer type NONE";
    }

    if (!success)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image buffer type not supported";
    }
}

IAllocator &ImageWrapData::alloc() const
{
    return GetDefaultAllocator();
}

Size2D ImageWrapData::size() const
{
    if (m_data.bufferType == NVCV_IMAGE_BUFFER_PITCH_DEVICE)
    {
        return {m_data.buffer.pitch.planes[0].width, m_data.buffer.pitch.planes[0].height};
    }
    else
    {
        NVCV_ASSERT(m_data.bufferType == NVCV_IMAGE_BUFFER_NONE);
        return {0, 0};
    }
}

ImageFormat ImageWrapData::format() const
{
    return ImageFormat{m_data.format};
}

void ImageWrapData::exportData(NVCVImageData &data) const
{
    data = m_data;
}

NVCVTypeImage ImageWrapData::type() const
{
    return NVCV_TYPE_IMAGE_WRAPDATA;
}

void ImageWrapData::doCleanup() noexcept
{
    if (m_cleanup && m_data.bufferType != NVCV_IMAGE_BUFFER_NONE)
    {
        m_cleanup(m_ctxCleanup, &m_data);
    }
}

} // namespace nv::cv::priv
