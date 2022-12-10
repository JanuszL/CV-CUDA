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
 * @file OpBilateralFilter.hpp
 *
 * @brief Defines the public C++ Class for the BilateralFilter operation.
 * @defgroup NVCV_CPP_ALGORITHM_BILATERAL_FILTER BilateralFilter
 * @{
 */

#ifndef NVCV_OP_BILATERAL_FILTER_HPP
#define NVCV_OP_BILATERAL_FILTER_HPP

#include "IOperator.hpp"
#include "OpBilateralFilter.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class BilateralFilter final : public IOperator
{
public:
    explicit BilateralFilter();

    ~BilateralFilter();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, int diameter, float sigmaColor,
                    float sigmaSpace, NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline BilateralFilter::BilateralFilter()
{
    cv::detail::CheckThrow(nvcvopBilateralFilterCreate(&m_handle));
    assert(m_handle);
}

inline BilateralFilter::~BilateralFilter()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void BilateralFilter::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, int diameter,
                                        float sigmaColor, float sigmaSpace, NVCVBorderType borderMode)
{
    cv::detail::CheckThrow(nvcvopBilateralFilterSubmit(m_handle, stream, in.handle(), out.handle(), diameter,
                                                       sigmaColor, sigmaSpace, borderMode));
}

inline NVCVOperatorHandle BilateralFilter::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_BILATERAL_FILTER_HPP
