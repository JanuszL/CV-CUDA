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

/**
 * @file OpCopyMakeBorder.hpp
 *
 * @brief Defines the public C++ class for the copy make border operation.
 * @defgroup NVCV_CPP_ALGORITHM_COPYMAKEBORDER Copy make border
 * @{
 */

#ifndef NVCV_OP_COPYMAKEBORDER_HPP
#define NVCV_OP_COPYMAKEBORDER_HPP

#include "IOperator.hpp"
#include "OpCopyMakeBorder.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class CopyMakeBorder final : public IOperator
{
public:
    explicit CopyMakeBorder();

    ~CopyMakeBorder();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, int32_t top, int32_t left,
                    NVCVBorderType borderMode, const float4 borderValue);
    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out, cv::ITensor &top,
                    cv::ITensor &left, NVCVBorderType borderMode, const float4 borderValue);
    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::ITensor &out, cv::ITensor &top,
                    cv::ITensor &left, NVCVBorderType borderMode, const float4 borderValue);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline CopyMakeBorder::CopyMakeBorder()
{
    cv::detail::CheckThrow(nvcvopCopyMakeBorderCreate(&m_handle));
    assert(m_handle);
}

inline CopyMakeBorder::~CopyMakeBorder()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void CopyMakeBorder::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, int32_t top,
                                       int32_t left, NVCVBorderType borderMode, const float4 borderValue)
{
    cv::detail::CheckThrow(
        nvcvopCopyMakeBorderSubmit(m_handle, stream, in.handle(), out.handle(), top, left, borderMode, borderValue));
}

inline void CopyMakeBorder::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                                       cv::ITensor &top, cv::ITensor &left, NVCVBorderType borderMode,
                                       const float4 borderValue)
{
    cv::detail::CheckThrow(nvcvopCopyMakeBorderVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), top.handle(),
                                                              left.handle(), borderMode, borderValue));
}

inline void CopyMakeBorder::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::ITensor &out,
                                       cv::ITensor &top, cv::ITensor &left, NVCVBorderType borderMode,
                                       const float4 borderValue)
{
    cv::detail::CheckThrow(nvcvopCopyMakeBorderVarShapeStackSubmit(
        m_handle, stream, in.handle(), out.handle(), top.handle(), left.handle(), borderMode, borderValue));
}

inline NVCVOperatorHandle CopyMakeBorder::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_COPYMAKEBORDER_HPP
