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

#ifndef NVCV_IMAGEBATCH_HPP
#define NVCV_IMAGEBATCH_HPP

#include "IImageBatch.hpp"
#include "ImageBatchData.hpp"

namespace nv { namespace cv {

// ImageBatchWrapHandle definition -------------------------------------
// Refers to an external NVCVImageBatch handle. It doesn't own it.
class ImageBatchWrapHandle : public virtual IImageBatch
{
public:
    ImageBatchWrapHandle(const ImageBatchWrapHandle &that);

    explicit ImageBatchWrapHandle(NVCVImageBatchHandle handle);

protected:
    mutable IImageBatchData *m_ptrData;

    const IImageBatchData *doExportData(CUstream stream) const override;

    IAllocator &doGetAlloc() const override;

    NVCVImageBatchHandle doGetHandle() const override;

private:
    NVCVImageBatchHandle m_handle;

    // Only one leaf, we can use an optional for now.
    mutable detail::Optional<ImageBatchVarShapeDataDevicePitch> m_data;

    mutable detail::Optional<AllocatorWrapHandle> m_alloc;

    int32_t     doGetCapacity() const override;
    int32_t     doGetSize() const override;
    ImageFormat doGetFormat() const override;
};

// ImageBatch varshape definition -------------------------------------
class ImageBatchVarShape
    : public IImageBatchVarShape
    , private ImageBatchWrapHandle
{
public:
    using Requirements = NVCVImageBatchVarShapeRequirements;
    static Requirements CalcRequirements(int32_t capacity, ImageFormat fmt);

    ImageBatchVarShape(const ImageBatchVarShape &) = delete;

    explicit ImageBatchVarShape(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit ImageBatchVarShape(int32_t capacity, ImageFormat fmt, IAllocator *alloc = nullptr);
    ~ImageBatchVarShape();

private:
    NVCVImageBatchStorage m_storage;
    IAllocator           *m_alloc;

    IAllocator &doGetAlloc() const override;

    void doPushBack(std::function<NVCVImageHandle()> &&cb) override;
    void doPushBack(const IImage &img) override;
    void doPopBack(int32_t imgCount) override;
    void doClear() override;

    NVCVImageHandle doGetImage(int32_t idx) const override;
};

// ImageBatchWrapHandle implementation -------------------------------------

inline ImageBatchWrapHandle::ImageBatchWrapHandle(const ImageBatchWrapHandle &that)
    : m_handle(that.m_handle)
{
}

inline ImageBatchWrapHandle::ImageBatchWrapHandle(NVCVImageBatchHandle handle)
    : m_handle(handle)
{
}

inline NVCVImageBatchHandle ImageBatchWrapHandle::doGetHandle() const
{
    return m_handle;
}

inline int32_t ImageBatchWrapHandle::doGetSize() const
{
    int32_t out;
    detail::CheckThrow(nvcvImageBatchGetSize(m_handle, &out));
    return out;
}

inline int32_t ImageBatchWrapHandle::doGetCapacity() const
{
    int32_t out;
    detail::CheckThrow(nvcvImageBatchGetCapacity(m_handle, &out));
    return out;
}

inline ImageFormat ImageBatchWrapHandle::doGetFormat() const
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvImageBatchGetFormat(m_handle, &out));
    return ImageFormat{out};
}

inline IAllocator &ImageBatchWrapHandle::doGetAlloc() const
{
    if (!m_alloc)
    {
        NVCVAllocatorHandle halloc;
        detail::CheckThrow(nvcvImageBatchGetAllocator(m_handle, &halloc));
        m_alloc.emplace(halloc);
    }

    return *m_alloc;
}

inline const IImageBatchData *ImageBatchWrapHandle::doExportData(CUstream stream) const
{
    NVCVImageBatchData batchData;
    detail::CheckThrow(nvcvImageBatchExportData(m_handle, stream, &batchData));

    assert(batchData.bufferType == NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_DEVICE_PITCH);

    m_data.emplace(ImageFormat{batchData.format}, batchData.buffer.varShapePitch);

    return &*m_data;
}

// ImageBatchVarShape implementation -------------------------------------

inline auto ImageBatchVarShape::CalcRequirements(int32_t capacity, ImageFormat fmt) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageBatchVarShapeCalcRequirements(capacity, fmt, &reqs));
    return reqs;
}

inline ImageBatchVarShape::ImageBatchVarShape(const Requirements &reqs, IAllocator *alloc)
    : ImageBatchWrapHandle(
        [&]
        {
            NVCVImageBatchHandle handle;
            detail::CheckThrow(
                nvcvImageBatchVarShapeConstruct(&reqs, alloc ? alloc->handle() : nullptr, &m_storage, &handle));
            return handle;
        }())
    , m_alloc(alloc)
{
}

inline ImageBatchVarShape::ImageBatchVarShape(int32_t capacity, ImageFormat fmt, IAllocator *alloc)
    : ImageBatchVarShape(CalcRequirements(capacity, fmt), alloc)
{
}

inline ImageBatchVarShape::~ImageBatchVarShape()
{
    // TODO: we're destroying the NVCVImageBatch *before* ImageBatchWrapHandle is
    // destroyed, which isn't advisable. But since we know its destructor won't
    // touch the handle, it's ok, or else we'd have to use some sort of
    // base-from-member idiom to get the destruction order right.
    nvcvImageBatchDestroy(ImageBatchWrapHandle::doGetHandle());
}

inline IAllocator &ImageBatchVarShape::doGetAlloc() const
{
    if (m_alloc != nullptr)
    {
        return *m_alloc;
    }
    else
    {
        return ImageBatchWrapHandle::doGetAlloc();
    }
}

inline void ImageBatchVarShape::doPushBack(std::function<NVCVImageHandle()> &&cb)
{
    static auto cbpriv = [](void *ctx) -> NVCVImageHandle
    {
        auto *cb = reinterpret_cast<std::function<NVCVImageHandle()> *>(ctx);
        assert(cb != nullptr);

        return (*cb)();
    };

    detail::CheckThrow(nvcvImageBatchVarShapePushImagesCallback(ImageBatchWrapHandle::doGetHandle(), cbpriv, &cb));
}

inline void ImageBatchVarShape::doPushBack(const IImage &img)
{
    NVCVImageHandle himg = img.handle();
    detail::CheckThrow(nvcvImageBatchVarShapePushImages(ImageBatchWrapHandle::doGetHandle(), &himg, 1));
}

inline void ImageBatchVarShape::doPopBack(int32_t imgCount)
{
    detail::CheckThrow(nvcvImageBatchVarShapePopImages(ImageBatchWrapHandle::doGetHandle(), imgCount));
}

inline void ImageBatchVarShape::doClear()
{
    detail::CheckThrow(nvcvImageBatchVarShapeClear(ImageBatchWrapHandle::doGetHandle()));
}

inline NVCVImageHandle ImageBatchVarShape::doGetImage(int32_t idx) const
{
    NVCVImageHandle himg;
    detail::CheckThrow(nvcvImageBatchVarShapeGetImages(ImageBatchWrapHandle::doGetHandle(), idx, &himg, 1));
    return himg;
}

}} // namespace nv::cv

#endif // NVCV_IMAGEBATCH_HPP
