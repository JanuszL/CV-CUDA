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
    mutable detail::Optional<ImageBatchVarShapeDataPitchDevice> m_data;

    mutable detail::Optional<AllocatorWrapHandle> m_alloc;

    int32_t     doGetCapacity() const override;
    int32_t     doGetNumImages() const override;
    ImageFormat doGetFormat() const override;
};

// ImageBatchVarShapeWrapHandle definition -------------------------------------
// Refers to an external varshape NVCVImageBatch handle. It doesn't own it.
class ImageBatchVarShapeWrapHandle
    : public virtual IImageBatchVarShape
    , private ImageBatchWrapHandle
{
public:
    ImageBatchVarShapeWrapHandle(const ImageBatchVarShapeWrapHandle &that);

    explicit ImageBatchVarShapeWrapHandle(NVCVImageBatchHandle handle);

    const IImageBatchVarShapeData *exportData(CUstream stream) const;

protected:
    using ImageBatchWrapHandle::doGetAlloc;

private:
    void doPushBack(std::function<NVCVImageHandle()> &&cb) override;
    void doPushBack(const IImage &img) override;
    void doPopBack(int32_t imgCount) override;
    void doClear() override;

    NVCVImageHandle doGetImage(int32_t idx) const override;
};

// ImageBatch varshape definition -------------------------------------
class ImageBatchVarShape final
    : public virtual IImageBatchVarShape
    , private ImageBatchVarShapeWrapHandle
{
public:
    using Requirements = NVCVImageBatchVarShapeRequirements;
    static Requirements CalcRequirements(int32_t capacity, ImageFormat fmt);

    ImageBatchVarShape(const ImageBatchVarShape &) = delete;

    explicit ImageBatchVarShape(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit ImageBatchVarShape(int32_t capacity, ImageFormat fmt, IAllocator *alloc = nullptr);
    ~ImageBatchVarShape();

    using IImageBatch::exportData;

private:
    NVCVImageBatchStorage m_storage;
    IAllocator           *m_alloc;

    IAllocator &doGetAlloc() const override;
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

inline int32_t ImageBatchWrapHandle::doGetNumImages() const
{
    int32_t out;
    detail::CheckThrow(nvcvImageBatchGetNumImages(m_handle, &out));
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

    if (batchData.bufferType != NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_PITCH_DEVICE)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION,
                        "Image batch data cannot be exported, buffer type not supported");
    }

    m_data.emplace(ImageFormat{batchData.format}, batchData.numImages, batchData.buffer.varShapePitch);

    return &*m_data;
}

// ImageBatchVarShapeWrapHandle implementation -------------------------------------

inline ImageBatchVarShapeWrapHandle::ImageBatchVarShapeWrapHandle(const ImageBatchVarShapeWrapHandle &that)
    : ImageBatchWrapHandle(that.handle())
{
}

inline ImageBatchVarShapeWrapHandle::ImageBatchVarShapeWrapHandle(NVCVImageBatchHandle handle)
    : ImageBatchWrapHandle(handle)
{
    NVCVTypeImageBatch type;
    detail::CheckThrow(nvcvImageBatchGetType(handle, &type));
    if (type != NVCV_TYPE_IMAGEBATCH_VARSHAPE)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image batch handle doesn't correspond to a varshape object");
    }
}

inline const IImageBatchVarShapeData *ImageBatchVarShapeWrapHandle::exportData(CUstream stream) const
{
    return static_cast<const IImageBatchVarShapeData *>(ImageBatchVarShapeWrapHandle::doExportData(stream));
}

inline void ImageBatchVarShapeWrapHandle::doPushBack(std::function<NVCVImageHandle()> &&cb)
{
    static auto cbpriv = [](void *ctx) -> NVCVImageHandle
    {
        auto *cb = reinterpret_cast<std::function<NVCVImageHandle()> *>(ctx);
        assert(cb != nullptr);

        return (*cb)();
    };

    detail::CheckThrow(nvcvImageBatchVarShapePushImagesCallback(ImageBatchWrapHandle::doGetHandle(), cbpriv, &cb));
}

inline void ImageBatchVarShapeWrapHandle::doPushBack(const IImage &img)
{
    NVCVImageHandle himg = img.handle();
    detail::CheckThrow(nvcvImageBatchVarShapePushImages(ImageBatchWrapHandle::doGetHandle(), &himg, 1));
}

inline void ImageBatchVarShapeWrapHandle::doPopBack(int32_t imgCount)
{
    detail::CheckThrow(nvcvImageBatchVarShapePopImages(ImageBatchWrapHandle::doGetHandle(), imgCount));
}

inline void ImageBatchVarShapeWrapHandle::doClear()
{
    detail::CheckThrow(nvcvImageBatchVarShapeClear(ImageBatchWrapHandle::doGetHandle()));
}

inline NVCVImageHandle ImageBatchVarShapeWrapHandle::doGetImage(int32_t idx) const
{
    NVCVImageHandle himg;
    detail::CheckThrow(nvcvImageBatchVarShapeGetImages(ImageBatchWrapHandle::doGetHandle(), idx, &himg, 1));
    return himg;
}

// ImageBatchVarShape implementation -------------------------------------

inline auto ImageBatchVarShape::CalcRequirements(int32_t capacity, ImageFormat fmt) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageBatchVarShapeCalcRequirements(capacity, fmt, &reqs));
    return reqs;
}

inline ImageBatchVarShape::ImageBatchVarShape(const Requirements &reqs, IAllocator *alloc)
    : ImageBatchVarShapeWrapHandle(
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
    nvcvImageBatchDestroy(this->handle());
}

inline IAllocator &ImageBatchVarShape::doGetAlloc() const
{
    if (m_alloc != nullptr)
    {
        return *m_alloc;
    }
    else
    {
        return ImageBatchVarShapeWrapHandle::doGetAlloc();
    }
}

}} // namespace nv::cv

#endif // NVCV_IMAGEBATCH_HPP
