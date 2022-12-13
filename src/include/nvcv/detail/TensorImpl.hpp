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

#ifndef NVCV_TENSOR_IMPL_HPP
#define NVCV_TENSOR_IMPL_HPP

#ifndef NVCV_TENSOR_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

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
{
    detail::CheckThrow(nvcvTensorConstruct(&reqs, alloc ? alloc->handle() : nullptr, &m_handle));
}

inline Tensor::Tensor(int numImages, Size2D imgSize, ImageFormat fmt, IAllocator *alloc)
    : Tensor(CalcRequirements(numImages, imgSize, fmt), alloc)
{
}

inline Tensor::Tensor(const TensorShape &shape, PixelType dtype, IAllocator *alloc)
    : Tensor(CalcRequirements(shape, dtype), alloc)
{
}

inline NVCVTensorHandle Tensor::doGetHandle() const
{
    return m_handle;
}

inline Tensor::~Tensor()
{
    nvcvTensorDestroy(m_handle);
}

// TensorWrapData implementation -------------------------------------

inline TensorWrapData::TensorWrapData(const ITensorData &data, std::function<TensorDataCleanupFunc> cleanup)
    : m_cleanup(std::move(cleanup))
{
    detail::CheckThrow(nvcvTensorWrapDataConstruct(&data.cdata(), m_cleanup ? &doCleanup : nullptr, this, &m_handle));
}

inline TensorWrapData::~TensorWrapData()
{
    nvcvTensorDestroy(m_handle);
}

inline NVCVTensorHandle TensorWrapData::doGetHandle() const
{
    return m_handle;
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

// TensorWrapImage implementation -------------------------------------

inline TensorWrapImage::TensorWrapImage(const IImage &img)
{
    detail::CheckThrow(nvcvTensorWrapImageConstruct(img.handle(), &m_handle));
}

inline TensorWrapImage::~TensorWrapImage()
{
    nvcvTensorDestroy(m_handle);
}

inline NVCVTensorHandle TensorWrapImage::doGetHandle() const
{
    return m_handle;
}

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

}} // namespace nv::cv

#endif // NVCV_TENSOR_IMPL_HPP
