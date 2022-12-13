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

#ifndef NVCV_IMAGE_IMPL_HPP
#define NVCV_IMAGE_IMPL_HPP

#ifndef NVCV_IMAGE_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// Image implementation -------------------------------------

inline auto Image::CalcRequirements(const Size2D &size, ImageFormat fmt, const MemAlignment &bufAlign) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageCalcRequirements(size.w, size.h, fmt, bufAlign.baseAddr(), bufAlign.rowAddr(), &reqs));
    return reqs;
}

inline Image::Image(const Requirements &reqs, IAllocator *alloc)
{
    detail::CheckThrow(nvcvImageConstruct(&reqs, alloc ? alloc->handle() : nullptr, &m_handle));
}

inline Image::Image(const Size2D &size, ImageFormat fmt, IAllocator *alloc, const MemAlignment &bufAlign)
    : Image(CalcRequirements(size, fmt, bufAlign), alloc)
{
}

inline Image::~Image()
{
    nvcvImageDestroy(m_handle);
}

inline NVCVImageHandle Image::doGetHandle() const
{
    return m_handle;
}

// ImageWrapData implementation -------------------------------------

inline ImageWrapData::ImageWrapData(const IImageData &data, std::function<ImageDataCleanupFunc> cleanup)
    : m_cleanup(std::move(cleanup))
{
    detail::CheckThrow(nvcvImageWrapDataConstruct(&data.cdata(), m_cleanup ? &doCleanup : nullptr, this, &m_handle));
}

inline ImageWrapData::~ImageWrapData()
{
    nvcvImageDestroy(m_handle);
}

inline NVCVImageHandle ImageWrapData::doGetHandle() const
{
    return m_handle;
}

inline void ImageWrapData::doCleanup(void *ctx, const NVCVImageData *data)
{
    assert(data != nullptr);

    auto *this_ = reinterpret_cast<ImageWrapData *>(ctx);
    assert(this_ != nullptr);

    // exportData refers to 'data'
    const IImageData *imgData = this_->exportData();
    assert(imgData != nullptr);

    assert(this_->m_cleanup != nullptr);
    this_->m_cleanup(*imgData);
}

// ImageWrapHandle implementation -------------------------------------

inline ImageWrapHandle::ImageWrapHandle(NVCVImageHandle handle)
    : m_handle(handle)
{
    assert(handle != nullptr);
}

inline ImageWrapHandle::ImageWrapHandle(const ImageWrapHandle &that)
    : m_handle(that.m_handle)
{
}

inline NVCVImageHandle ImageWrapHandle::doGetHandle() const
{
    return m_handle;
}

}} // namespace nv::cv

#endif // NVCV_IMAGE_IMPL_HPP
