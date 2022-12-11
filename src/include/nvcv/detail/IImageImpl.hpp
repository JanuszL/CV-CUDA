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

#ifndef NVCV_IIMAGE_IMPL_HPP
#define NVCV_IIMAGE_IMPL_HPP

#ifndef NVCV_IIMAGE_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

inline IImage::IImage()
    : m_cacheDataPtr(nullptr)
{
}

inline IImage::~IImage()
{
    if (m_cacheDataPtr != nullptr)
    {
        m_cacheDataPtr->~IImageData();
    }
}

inline NVCVImageHandle IImage::handle() const
{
    NVCVImageHandle h = doGetHandle();
    assert(h != nullptr && "Post-condition failed");
    return h;
}

inline Size2D IImage::size() const
{
    Size2D out;
    detail::CheckThrow(nvcvImageGetSize(this->handle(), &out.w, &out.h));
    return out;
}

inline ImageFormat IImage::format() const
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvImageGetFormat(this->handle(), &out));
    return ImageFormat{out};
}

inline const IImageData *IImage::exportData() const
{
    NVCVImageData imgData;
    detail::CheckThrow(nvcvImageExportData(this->handle(), &imgData));

    if (m_cacheDataPtr != nullptr)
    {
        m_cacheDataPtr->~IImageData();
        m_cacheDataPtr = nullptr;
    }

    switch (imgData.bufferType)
    {
    case NVCV_IMAGE_BUFFER_PITCH_HOST:
    case NVCV_IMAGE_BUFFER_NONE:
        break; // will return nullptr as per current semantics

    case NVCV_IMAGE_BUFFER_PITCH_DEVICE:
        m_cacheDataPtr
            = ::new (&m_cacheDataArena) ImageDataPitchDevice(ImageFormat{imgData.format}, imgData.buffer.pitch);
        break;

    case NVCV_IMAGE_BUFFER_CUDA_ARRAY:
        m_cacheDataPtr
            = ::new (&m_cacheDataArena) ImageDataCudaArray(ImageFormat{imgData.format}, imgData.buffer.cudaarray);
        break;
    }

    return m_cacheDataPtr;
}

}} // namespace nv::cv

#endif // NVCV_IIMAGE_IMPL_HPP
