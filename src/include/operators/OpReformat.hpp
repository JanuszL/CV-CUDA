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
 * @file OpReformat.hpp
 *
 * @brief Defines the public C++ Class for the reformat operation.
 * @defgroup NVCV_CPP_ALGORITHM_REFORMAT Reformat
 * @{
 */

#ifndef NVCV_OP_REFORMAT_HPP
#define NVCV_OP_REFORMAT_HPP

#include "IOperator.hpp"
#include "OpReformat.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Reformat final : public IOperator
{
public:
    explicit Reformat();

    ~Reformat();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Reformat::Reformat()
{
    cv::detail::CheckThrow(nvcvopReformatCreate(&m_handle));
    assert(m_handle);
}

inline Reformat::~Reformat()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Reformat::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out)
{
    cv::detail::CheckThrow(nvcvopReformatSubmit(m_handle, stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle Reformat::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_REFORMAT_HPP
