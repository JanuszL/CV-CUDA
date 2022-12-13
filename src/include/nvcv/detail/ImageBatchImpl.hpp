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

#ifndef NVCV_IMAGEBATCH_IMPL_HPP
#define NVCV_IMAGEBATCH_IMPL_HPP

#ifndef NVCV_IMAGEBATCH_IMPL_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// ImageBatchVarShape implementation -------------------------------------

inline auto ImageBatchVarShape::CalcRequirements(int32_t capacity) -> Requirements
{
    Requirements reqs;
    detail::CheckThrow(nvcvImageBatchVarShapeCalcRequirements(capacity, &reqs));
    return reqs;
}

inline ImageBatchVarShape::ImageBatchVarShape(const Requirements &reqs, IAllocator *alloc)
{
    detail::CheckThrow(nvcvImageBatchVarShapeConstruct(&reqs, alloc ? alloc->handle() : nullptr, &m_handle));
}

inline ImageBatchVarShape::ImageBatchVarShape(int32_t capacity, IAllocator *alloc)
    : ImageBatchVarShape(CalcRequirements(capacity), alloc)
{
}

inline ImageBatchVarShape::~ImageBatchVarShape()
{
    nvcvImageBatchDestroy(m_handle);
}

inline NVCVImageBatchHandle ImageBatchVarShape::doGetHandle() const
{
    return m_handle;
}

// ImageBatchWrapHandle implementation -------------------------------------

inline ImageBatchWrapHandle::ImageBatchWrapHandle(NVCVImageBatchHandle handle)
    : m_handle(handle)
{
}

inline ImageBatchWrapHandle::ImageBatchWrapHandle(const ImageBatchWrapHandle &that)
    : m_handle(that.m_handle)
{
}

inline NVCVImageBatchHandle ImageBatchWrapHandle::doGetHandle() const
{
    return m_handle;
}

// ImageBatchVarShapeWrapHandle implementation -------------------------------------

inline ImageBatchVarShapeWrapHandle::ImageBatchVarShapeWrapHandle(NVCVImageBatchHandle handle)
    : m_handle(handle)
{
    NVCVTypeImageBatch type;
    detail::CheckThrow(nvcvImageBatchGetType(m_handle, &type));
    if (type != NVCV_TYPE_IMAGEBATCH_VARSHAPE)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Image batch handle doesn't correspond to a varshape object");
    }
}

inline ImageBatchVarShapeWrapHandle::ImageBatchVarShapeWrapHandle(const ImageBatchVarShapeWrapHandle &that)
    : m_handle(that.m_handle)
{
}

inline NVCVImageBatchHandle ImageBatchVarShapeWrapHandle::doGetHandle() const
{
    return m_handle;
}

}} // namespace nv::cv

#endif // NVCV_IMAGEBATCH_IMPL_HPP
