/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
