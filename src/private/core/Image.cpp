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

} // namespace nv::cv::priv
