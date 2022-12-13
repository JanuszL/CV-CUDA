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

// ImageBatch varshape definition -------------------------------------
class ImageBatchVarShape : public IImageBatchVarShape
{
public:
    using Requirements = NVCVImageBatchVarShapeRequirements;
    static Requirements CalcRequirements(int32_t capacity);

    explicit ImageBatchVarShape(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit ImageBatchVarShape(int32_t capacity, IAllocator *alloc = nullptr);
    ~ImageBatchVarShape();

    ImageBatchVarShape(const ImageBatchVarShape &) = delete;

private:
    NVCVImageBatchHandle doGetHandle() const final;

    NVCVImageBatchHandle m_handle;
};

// ImageBatchWrapHandle definition -------------------------------------
// Refers to an external NVCVImageBatch handle. It doesn't own it.
class ImageBatchWrapHandle : public IImageBatch
{
public:
    explicit ImageBatchWrapHandle(NVCVImageBatchHandle handle);

    ImageBatchWrapHandle(const ImageBatchWrapHandle &that);

private:
    NVCVImageBatchHandle doGetHandle() const final;

    NVCVImageBatchHandle m_handle;
};

// ImageBatchVarShapeWrapHandle definition -------------------------------------
// Refers to an external varshape NVCVImageBatch handle. It doesn't own it.
class ImageBatchVarShapeWrapHandle : public IImageBatchVarShape
{
public:
    explicit ImageBatchVarShapeWrapHandle(NVCVImageBatchHandle handle);

    ImageBatchVarShapeWrapHandle(const ImageBatchVarShapeWrapHandle &that);

private:
    NVCVImageBatchHandle doGetHandle() const final;

    NVCVImageBatchHandle m_handle;
};

}} // namespace nv::cv

#include "detail/ImageBatchImpl.hpp"

#endif // NVCV_IMAGEBATCH_HPP
