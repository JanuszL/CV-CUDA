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

/**
 * @file OpCopyMakeBorder.hpp
 *
 * @brief Defines the public C++ class for the copy make border operation.
 * @defgroup NVCV_CPP_ALGORITHM_COPYMAKEBORDER Copy make border
 * @{
 */

#ifndef NVCV_OP_PADANDSTACK_HPP
#define NVCV_OP_PADANDSTACK_HPP

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

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, int top, int left,
                    NVCVBorderType borderMode, const float4 borderValue);

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

inline void CopyMakeBorder::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, int top, int left,
                                       NVCVBorderType borderMode, const float4 borderValue)
{
    cv::detail::CheckThrow(
        nvcvopCopyMakeBorderSubmit(m_handle, stream, in.handle(), out.handle(), top, left, borderMode, borderValue));
}

inline NVCVOperatorHandle CopyMakeBorder::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_COPYMAKEBORDER_HPP
