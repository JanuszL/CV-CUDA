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

#ifndef NVCV_IMAGE_HPP
#define NVCV_IMAGE_HPP

#include "IImage.hpp"
#include "ImageData.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"
#include "alloc/AllocatorWrapHandle.hpp"
#include "detail/Optional.hpp"

#include <type_traits>

namespace nv { namespace cv {

// ImageWrapHandle definition -------------------------------------
// Refers to an external NVCVImage handle. It doesn't own it.
class ImageWrapHandle : public virtual IImage
{
public:
    ImageWrapHandle(const ImageWrapHandle &that);

    explicit ImageWrapHandle(NVCVImage handle);
    ~ImageWrapHandle();

protected:
    mutable IImageData *m_ptrData;

    IAllocator       &doGetAlloc() const override;
    const IImageData *doExportData() const override;
    NVCVImage         doGetHandle() const override;

private:
    NVCVImage m_handle;

    // Where the concrete class for exported image data will be allocated
    // Should be an std::variant in C++17.
    union Arena
    {
        ImageDataCudaArray   cudaArray;
        ImageDataDevicePitch devPitch;
    };

    mutable std::aligned_storage<sizeof(Arena), alignof(Arena)>::type m_dataArena;

    mutable detail::Optional<AllocatorWrapHandle> m_alloc;

    Size2D      doGetSize() const override;
    ImageFormat doGetFormat() const override;
};

// Image definition -------------------------------------
// Image allocated by cv-cuda
class Image final
    : public virtual IImage
    , private ImageWrapHandle
{
public:
    using Requirements = NVCVImageRequirements;
    static Requirements CalcRequirements(const Size2D &size, ImageFormat fmt);

    Image(const Image &) = delete;

    explicit Image(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit Image(const Size2D &size, ImageFormat fmt, IAllocator *alloc = nullptr);
    ~Image();

private:
    IAllocator *m_alloc;

    const IImageData *doExportData() const override;
    IAllocator       &doGetAlloc() const override;
};

// ImageWrapData definition -------------------------------------
// Image that wraps an image data allocated outside cv-cuda
class ImageWrapData final
    : public IImageWrapData
    , private ImageWrapHandle
{
public:
    explicit ImageWrapData(IAllocator *alloc = nullptr);
    explicit ImageWrapData(const IImageData &data, IAllocator *alloc = nullptr);
    explicit ImageWrapData(const IImageData &data, std::function<ImageDataCleanupFunc> cleanup,
                           IAllocator *alloc = nullptr);

    ~ImageWrapData();

private:
    IAllocator *m_alloc;

    std::function<ImageDataCleanupFunc> m_cleanup;
    static void                         doCleanup(void *ctx, const NVCVImageData *data);

    void        doResetData(const IImageData *data) override;
    void        doResetDataAndCleanup(const IImageData *data, std::function<ImageDataCleanupFunc> cleanup) override;
    IAllocator &doGetAlloc() const override;
    const IImageData *doExportData() const override;
};

// ImageWrapHandle implementation -------------------------------------

inline ImageWrapHandle::ImageWrapHandle(const ImageWrapHandle &that)
    : m_ptrData(nullptr) // will be fetched again when needed
    , m_handle(that.m_handle)
{
}

inline ImageWrapHandle::ImageWrapHandle(NVCVImage handle)
    : m_ptrData(nullptr)
    , m_handle(handle)
{
}

inline ImageWrapHandle::~ImageWrapHandle()
{
    if (m_ptrData != nullptr)
    {
        m_ptrData->~IImageData();
    }
}

inline NVCVImage ImageWrapHandle::doGetHandle() const
{
    return m_handle;
}

inline Size2D ImageWrapHandle::doGetSize() const
{
    Size2D out;
    detail::CheckThrow(nvcvImageGetSize(m_handle, &out.w, &out.h));
    return out;
}

inline ImageFormat ImageWrapHandle::doGetFormat() const
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvImageGetFormat(m_handle, &out));
    return ImageFormat{out};
}

inline IAllocator &ImageWrapHandle::doGetAlloc() const
{
    if (!m_alloc)
    {
        NVCVAllocatorHandle halloc;
        detail::CheckThrow(nvcvImageGetAllocator(m_handle, &halloc));
        m_alloc.emplace(halloc);
    }

    return *m_alloc;
}

inline const IImageData *ImageWrapHandle::doExportData() const
{
    NVCVImageData imgData;
    detail::CheckThrow(nvcvImageExportData(m_handle, &imgData));

    if (m_ptrData != nullptr)
    {
        m_ptrData->~IImageData();
        m_ptrData = nullptr;
    }

    switch (imgData.bufferType)
    {
    case NVCV_IMAGE_BUFFER_NONE:
        break;

    case NVCV_IMAGE_BUFFER_DEVICE_PITCH:
        m_ptrData = ::new (&m_dataArena) ImageDataDevicePitch(ImageFormat{imgData.format}, imgData.buffer.pitch);
        break;

    case NVCV_IMAGE_BUFFER_CUDA_ARRAY:
        m_ptrData = ::new (&m_dataArena) ImageDataCudaArray(ImageFormat{imgData.format}, imgData.buffer.cudaarray);
        break;
    }

    return m_ptrData;
}

// Image implementation -------------------------------------

inline auto Image::CalcRequirements(const Size2D &size, ImageFormat fmt) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageCalcRequirements(size.w, size.h, fmt, &reqs));
    return reqs;
}

inline Image::Image(const Requirements &reqs, IAllocator *alloc)
    : ImageWrapHandle(
        [&]
        {
            NVCVImage handle;
            detail::CheckThrow(nvcvImageCreate(&reqs, alloc ? alloc->handle() : nullptr, &handle));
            return handle;
        }())
    , m_alloc(alloc)
{
}

inline Image::Image(const Size2D &size, ImageFormat fmt, IAllocator *alloc)
    : Image(CalcRequirements(size, fmt), alloc)
{
}

inline Image::~Image()
{
    // TODO: we're destroying the VPIImage *before* ImageWrapHandle is
    // destroyed, which isn't advisable. But since we know its destructor won't
    // touch the handle, it's ok, or else we'd have to use some sort of
    // base-from-member idiom to get the destruction order right.
    nvcvImageDestroy(ImageWrapHandle::doGetHandle());
}

inline IAllocator &Image::doGetAlloc() const
{
    if (m_alloc != nullptr)
    {
        return *m_alloc;
    }
    else
    {
        return ImageWrapHandle::doGetAlloc();
    }
}

inline const IImageData *Image::doExportData() const
{
    // Export data already fetched?
    if (m_ptrData != nullptr)
    {
        // export data of an Image object is immutable (both buffer and
        // metadata), so we can just return here what we previously fetched.
        return m_ptrData;
    }
    else
    {
        return ImageWrapHandle::doExportData();
    }
}

// ImageWrapData implementation -------------------------------------

inline ImageWrapData::ImageWrapData(IAllocator *alloc)
    : ImageWrapHandle(
        [&]
        {
            NVCVImage himg;
            detail::CheckThrow(nvcvImageCreateWrapData(nullptr,          // data
                                                       nullptr, nullptr, // cleanup and ctx
                                                       alloc ? alloc->handle() : nullptr, &himg));
            return himg;
        }())
    , m_alloc(alloc)
{
}

inline ImageWrapData::ImageWrapData(const IImageData &data, std::function<ImageDataCleanupFunc> cleanup,
                                    IAllocator *alloc)
    : ImageWrapHandle(
        [&]
        {
            NVCVImage himg;
            detail::CheckThrow(nvcvImageCreateWrapData(&data.cdata(), cleanup ? &doCleanup : nullptr, this,
                                                       alloc ? alloc->handle() : nullptr, &himg));
            return himg;
        }())
    , m_alloc(alloc)
    , m_cleanup(std::move(cleanup))
{
}

inline ImageWrapData::ImageWrapData(const IImageData &data, IAllocator *alloc)
    : ImageWrapData(data, nullptr, alloc)
{
}

inline ImageWrapData::~ImageWrapData()
{
    // TODO: we're destroying the VPIImage *before* ImageWrapHandle is
    // destroyed, which isn't advisable. But since we know its destructor won't
    // touch the handle, it's ok, or else we'd have to use some sort of
    // base-from-member idiom to get the destruction order right.
    nvcvImageDestroy(ImageWrapHandle::doGetHandle());
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

inline const IImageData *ImageWrapData::doExportData() const
{
    // Export data already fetched?
    if (m_ptrData != nullptr)
    {
        // export data of an Image object is immutable (both buffer and
        // metadata), so we can just return here what we previously fetched.
        return m_ptrData;
    }
    else
    {
        return ImageWrapHandle::doExportData();
    }
}

inline void ImageWrapData::doResetDataAndCleanup(const IImageData *data, std::function<ImageDataCleanupFunc> cleanup)
{
    detail::CheckThrow(nvcvImageWrapResetDataAndCleanup(this->handle(), data ? &data->cdata() : nullptr,
                                                        cleanup ? &doCleanup : nullptr, this));

    // Only set the new cleanup function after we call C API, as it'll end up calling the
    // current m_cleanup, if defined.
    m_cleanup = std::move(cleanup);
}

inline void ImageWrapData::doResetData(const IImageData *data)
{
    detail::CheckThrow(nvcvImageWrapResetData(this->handle(), data ? &data->cdata() : nullptr));
}

inline IAllocator &ImageWrapData::doGetAlloc() const
{
    if (m_alloc != nullptr)
    {
        return *m_alloc;
    }
    else
    {
        return ImageWrapHandle::doGetAlloc();
    }
}

}} // namespace nv::cv

#endif // NVCV_IMAGE_HPP
