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

#ifndef NVCV_TENSOR_HPP
#define NVCV_TENSOR_HPP

#include "ITensor.hpp"
#include "TensorData.hpp"

namespace nv { namespace cv {

// TensorWrapHandle definition -------------------------------------
// Refers to an external NVCVTensor handle. It doesn't own it.
class TensorWrapHandle : public virtual ITensor
{
public:
    TensorWrapHandle(const TensorWrapHandle &that);
    explicit TensorWrapHandle(NVCVTensorHandle handle);

protected:
    const ITensorData *doExportData() const override;
    IAllocator        &doGetAlloc() const override;
    NVCVTensorHandle   doGetHandle() const override;

    mutable detail::Optional<TensorDataPitchDevice> m_optData;

private:
    NVCVTensorHandle m_handle;

    mutable detail::Optional<AllocatorWrapHandle> m_alloc;

    int          doGetNumDim() const override;
    TensorShape  doGetShape() const override;
    PixelType    doGetDataType() const override;
    TensorLayout doGetLayout() const override;
};

// Tensor tensor definition -------------------------------------
class Tensor
    : public virtual ITensor
    , private TensorWrapHandle
{
public:
    using Requirements = NVCVTensorRequirements;
    static Requirements CalcRequirements(const TensorShape &shape, PixelType dtype);
    static Requirements CalcRequirements(int numImages, Size2D imgSize, ImageFormat fmt);

    Tensor(const Tensor &) = delete;

    explicit Tensor(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit Tensor(const TensorShape &shape, PixelType dtype, IAllocator *alloc = nullptr);
    explicit Tensor(int numImages, Size2D imgSize, ImageFormat fmt, IAllocator *alloc = nullptr);
    ~Tensor();

private:
    NVCVTensorStorage m_storage;
    IAllocator       *m_alloc;

    IAllocator &doGetAlloc() const override;
};

// TensorWrapData definition -------------------------------------

using TensorDataCleanupFunc = void(const ITensorData &);

class TensorWrapData
    : public virtual ITensor
    , private TensorWrapHandle
{
public:
    TensorWrapData(const TensorWrapData &) = delete;

    explicit TensorWrapData(const ITensorData &data, std::function<TensorDataCleanupFunc> cleanup = nullptr);
    ~TensorWrapData();

private:
    NVCVTensorStorage m_storage;

    std::function<TensorDataCleanupFunc> m_cleanup;

    static void doCleanup(void *ctx, const NVCVTensorData *data);

    const ITensorData *doExportData() const override;
};

// TensorWrapImage definition -------------------------------------

class TensorWrapImage
    : public virtual ITensor
    , private TensorWrapHandle
{
public:
    TensorWrapImage(const TensorWrapImage &) = delete;

    explicit TensorWrapImage(const IImage &mg);
    ~TensorWrapImage();

private:
    NVCVTensorStorage m_storage;
};

// TensorWrapHandle implementation -------------------------------------

inline TensorWrapHandle::TensorWrapHandle(const TensorWrapHandle &that)
    : m_handle(that.m_handle)
{
}

inline TensorWrapHandle::TensorWrapHandle(NVCVTensorHandle handle)
    : m_handle(handle)
{
}

inline NVCVTensorHandle TensorWrapHandle::doGetHandle() const
{
    return m_handle;
}

inline TensorLayout TensorWrapHandle::doGetLayout() const
{
    NVCVTensorLayout layout;
    detail::CheckThrow(nvcvTensorGetLayout(m_handle, &layout));
    return static_cast<TensorLayout>(layout);
}

inline int TensorWrapHandle::doGetNumDim() const
{
    int32_t ndim = 0;
    detail::CheckThrow(nvcvTensorGetShape(m_handle, &ndim, nullptr));
    return ndim;
}

inline TensorShape TensorWrapHandle::doGetShape() const
{
    int32_t ndim = 0;
    detail::CheckThrow(nvcvTensorGetShape(m_handle, &ndim, nullptr));

    NVCVTensorLayout layout;
    detail::CheckThrow(nvcvTensorGetLayout(m_handle, &layout));

    TensorShape::ShapeType shape(ndim);
    detail::CheckThrow(nvcvTensorGetShape(m_handle, &ndim, shape.begin()));
    return {shape, layout};
}

inline PixelType TensorWrapHandle::doGetDataType() const
{
    NVCVPixelType out;
    detail::CheckThrow(nvcvTensorGetDataType(m_handle, &out));
    return PixelType{out};
}

inline IAllocator &TensorWrapHandle::doGetAlloc() const
{
    if (!m_alloc)
    {
        NVCVAllocatorHandle halloc;
        detail::CheckThrow(nvcvTensorGetAllocator(m_handle, &halloc));
        m_alloc.emplace(halloc);
    }

    return *m_alloc;
}

inline const ITensorData *TensorWrapHandle::doExportData() const
{
    NVCVTensorData data;
    detail::CheckThrow(nvcvTensorExportData(m_handle, &data));

    assert(data.bufferType == NVCV_TENSOR_BUFFER_PITCH_DEVICE);

    m_optData.emplace(TensorShape(data.shape, data.ndim, data.layout), PixelType{data.dtype}, data.buffer.pitch);

    return &*m_optData;
}

// Tensor implementation -------------------------------------

inline auto Tensor::CalcRequirements(const TensorShape &shape, PixelType dtype) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvTensorCalcRequirements(shape.size(), &shape[0], dtype,
                                                  static_cast<NVCVTensorLayout>(shape.layout()), &reqs));
    return reqs;
}

inline auto Tensor::CalcRequirements(int numImages, Size2D imgSize, ImageFormat fmt) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvTensorCalcRequirementsForImages(numImages, imgSize.w, imgSize.h, fmt, &reqs));
    return reqs;
}

inline Tensor::Tensor(const Requirements &reqs, IAllocator *alloc)
    : TensorWrapHandle(
        [&]
        {
            NVCVTensorHandle handle;
            detail::CheckThrow(nvcvTensorConstruct(&reqs, alloc ? alloc->handle() : nullptr, &m_storage, &handle));
            return handle;
        }())
    , m_alloc(alloc)
{
}

inline Tensor::Tensor(int numImages, Size2D imgSize, ImageFormat fmt, IAllocator *alloc)
    : Tensor(CalcRequirements(numImages, imgSize, fmt), alloc)
{
}

inline Tensor::Tensor(const TensorShape &shape, PixelType dtype, IAllocator *alloc)
    : Tensor(CalcRequirements(shape, dtype), alloc)
{
}

inline Tensor::~Tensor()
{
    // TODO: we're destroying the NVCVTensor *before* TensorWrapHandle is
    // destroyed, which isn't advisable. But since we know its destructor won't
    // touch the handle, it's ok, or else we'd have to use some sort of
    // base-from-member idiom to get the destruction order right.
    nvcvTensorDestroy(TensorWrapHandle::doGetHandle());
}

inline IAllocator &Tensor::doGetAlloc() const
{
    if (m_alloc != nullptr)
    {
        return *m_alloc;
    }
    else
    {
        return TensorWrapHandle::doGetAlloc();
    }
}

// TensorWrapData implementation -------------------------------------

inline TensorWrapData::TensorWrapData(const ITensorData &data, std::function<TensorDataCleanupFunc> cleanup)
    : TensorWrapHandle(
        [&]
        {
            NVCVTensorHandle handle;
            detail::CheckThrow(
                nvcvTensorWrapDataConstruct(&data.cdata(), cleanup ? &doCleanup : nullptr, this, &m_storage, &handle));
            return handle;
        }())
    , m_cleanup(std::move(cleanup))
{
}

inline TensorWrapData::~TensorWrapData()
{
    // TODO: we're destroying the NVCVTensor *before* ImageWrapHandle is
    // destroyed, which isn't advisable. But since we know its destructor won't
    // touch the handle, it's ok, or else we'd have to use some sort of
    // base-from-member idiom to get the destruction order right.
    nvcvTensorDestroy(doGetHandle());
}

inline void TensorWrapData::doCleanup(void *ctx, const NVCVTensorData *data)
{
    assert(data != nullptr);

    auto *this_ = reinterpret_cast<TensorWrapData *>(ctx);
    assert(this_ != nullptr);

    // exportData refers to 'data'
    const ITensorData *batchData = this_->exportData();
    assert(batchData != nullptr);

    assert(this_->m_cleanup != nullptr);
    this_->m_cleanup(*batchData);
}

inline const ITensorData *TensorWrapData::doExportData() const
{
    // Export data already fetched?
    if (m_optData)
    {
        // export data of an Image object is immutable (both buffer and
        // metadata), so we can just return here what we previously fetched.
        return &*m_optData;
    }
    else
    {
        return TensorWrapHandle::doExportData();
    }
}

// TensorWrapImage implementation -------------------------------------

inline TensorWrapImage::TensorWrapImage(const IImage &img)
    : TensorWrapHandle(
        [&]
        {
            NVCVTensorHandle handle;
            detail::CheckThrow(nvcvTensorWrapImageConstruct(img.handle(), &m_storage, &handle));
            return handle;
        }())
{
}

inline TensorWrapImage::~TensorWrapImage()
{
    nvcvTensorDestroy(doGetHandle());
}

}} // namespace nv::cv

#endif // NVCV_TENSOR_HPP
