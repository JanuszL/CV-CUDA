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
 * @file OpPadAndStack.hpp
 *
 * @brief Defines the public C++ class for the pad and stack operation.
 * @defgroup NVCV_CPP_ALGORITHM_PADANDSTACK Pad and stack
 * @{
 */

#ifndef NVCV_OP_PADANDSTACK_HPP
#define NVCV_OP_PADANDSTACK_HPP

#include "IOperator.hpp"
#include "OpPadAndStack.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class PadAndStack final : public IOperator
{
public:
    explicit PadAndStack();

    ~PadAndStack();

    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::ITensor &out, cv::ITensor &left,
                    cv::ITensor &top, const NVCVBorderType borderMode, const float borderValue);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline PadAndStack::PadAndStack()
{
    cv::detail::CheckThrow(nvcvopPadAndStackCreate(&m_handle));
    assert(m_handle);
}

inline PadAndStack::~PadAndStack()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void PadAndStack::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::ITensor &out,
                                    cv::ITensor &left, cv::ITensor &top, const NVCVBorderType borderMode,
                                    const float borderValue)
{
    cv::detail::CheckThrow(nvcvopPadAndStackSubmit(m_handle, stream, in.handle(), out.handle(), left.handle(),
                                                   top.handle(), borderMode, borderValue));
}

inline NVCVOperatorHandle PadAndStack::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_PADANDSTACK_HPP
