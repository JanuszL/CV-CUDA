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
 * @file OpConverTo.hpp
 *
 * @brief Defines the public C++ Class for the ConvertTo operation.
 * @defgroup NVCV_CPP_ALGORITHM_CONVERT_TO ConvertTo
 * @{
 */

#ifndef NVCV_OP_CONVERT_TO_HPP
#define NVCV_OP_CONVERT_TO_HPP

#include "IOperator.hpp"
#include "OpConvertTo.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class ConvertTo final : public IOperator
{
public:
    explicit ConvertTo();

    ~ConvertTo();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const double alpha, const double beta);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline ConvertTo::ConvertTo()
{
    cv::detail::CheckThrow(nvcvopConvertToCreate(&m_handle));
    assert(m_handle);
}

inline ConvertTo::~ConvertTo()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void ConvertTo::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const double alpha,
                                  const double beta)
{
    cv::detail::CheckThrow(nvcvopConvertToSubmit(m_handle, stream, in.handle(), out.handle(), alpha, beta));
}

inline NVCVOperatorHandle ConvertTo::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_CONVERT_TO_HPP
