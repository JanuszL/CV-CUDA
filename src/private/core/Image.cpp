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

#include "Image.hpp"

#include "AllocInfo.hpp"
#include "IAllocator.hpp"

#include <fmt/PixelType.hpp>

namespace nv::cv::priv {

// Image implementation -------------------------------------------

Image::Image(Size2D size, ImageFormat fmt, IAllocator &alloc)
    : m_alloc{alloc}
    , m_size(size)
    , m_format(fmt)
{
    if (fmt.memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED, "Image with block-linear format is not currently supported.");
    }

    m_allocInfo = CalcAllocInfo(size, fmt);
    m_buffer    = m_alloc.allocDeviceMem(m_allocInfo.size, m_allocInfo.alignment);
    NVCV_ASSERT(m_buffer != nullptr);
}

Image::~Image()
{
    m_alloc.freeDeviceMem(m_buffer, m_allocInfo.size, m_allocInfo.alignment);
}

NVCVTypeImage Image::type() const
{
    return NVCV_TYPE_IMAGE;
}

Version Image::doGetVersion() const
{
    return CURRENT_VERSION;
}

Size2D Image::size() const
{
    return m_size;
}

ImageFormat Image::format() const
{
    return m_format;
}

IAllocator &Image::alloc() const
{
    return m_alloc;
}

void Image::exportData(NVCVImageData &data) const
{
    NVCV_ASSERT(this->format().memLayout() == NVCV_MEM_LAYOUT_PL);

    data.format     = m_format.value();
    data.bufferType = NVCV_IMAGE_BUFFER_DEVICE_PITCH;

    NVCVImageBufferPitch &buf = data.buffer.pitch;

    buf.numPlanes = m_allocInfo.planes.size();
    for (int p = 0; p < buf.numPlanes; ++p)
    {
        NVCVImagePlanePitch &plane = buf.planes[p];

        Size2D planeSize = m_format.planeSize(m_size, p);

        plane.width      = planeSize.w;
        plane.height     = planeSize.h;
        plane.pitchBytes = m_allocInfo.planes[p].rowPitchBytes;
        plane.buffer     = reinterpret_cast<std::byte *>(m_buffer) + m_allocInfo.planes[p].offsetBytes;
    }
}

// ImageWrap implementation -------------------------------------------

ImageWrapData::ImageWrapData(IAllocator &alloc)
    : m_alloc(alloc)
{
    m_data.bufferType = NVCV_IMAGE_BUFFER_NONE;
    m_data.format     = NVCV_IMAGE_FORMAT_NONE;
}

ImageWrapData::ImageWrapData(const NVCVImageData &data, IAllocator &alloc)
    : m_alloc(alloc)
{
    doValidateData(data);

    m_data = data;
}

void ImageWrapData::doValidateData(const NVCVImageData &data) const
{
    ImageFormat format{data.format};

    bool success = false;
    switch (data.bufferType)
    {
    case NVCV_IMAGE_BUFFER_DEVICE_PITCH:
        if (format.memLayout() != NVCV_MEM_LAYOUT_PL)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Image buffer type DEVICE_PITCH not consistent with image format " << format;
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
    return m_alloc;
}

Size2D ImageWrapData::size() const
{
    if (m_data.bufferType == NVCV_IMAGE_BUFFER_DEVICE_PITCH)
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
    return NVCV_TYPE_IMAGE_WRAP_DATA;
}

Version ImageWrapData::doGetVersion() const
{
    return CURRENT_VERSION;
}

void ImageWrapData::setData(const NVCVImageData &data)
{
    doValidateData(data);
    m_data = data;
}

void ImageWrapData::resetData()
{
    m_data.bufferType = NVCV_IMAGE_BUFFER_NONE;
    m_data.format     = NVCV_IMAGE_FORMAT_NONE;
}

} // namespace nv::cv::priv
