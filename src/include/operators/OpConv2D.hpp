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
 * @file OpConv2D.hpp
 *
 * @brief Defines the public C++ Class for the 2D convolution operation.
 * @defgroup NVCV_CPP_ALGORITHM_CONV2D 2D Convolution
 * @{
 */

#ifndef NVCV_OP_CONV2D_HPP
#define NVCV_OP_CONV2D_HPP

#include "IOperator.hpp"
#include "OpConv2D.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Conv2D final : public IOperator
{
public:
    explicit Conv2D();

    ~Conv2D();

    void operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out, cv::IImageBatch &kernel,
                    cv::ITensor &kernelAnchor, NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Conv2D::Conv2D()
{
    cv::detail::CheckThrow(nvcvopConv2DCreate(&m_handle));
    assert(m_handle);
}

inline Conv2D::~Conv2D()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Conv2D::operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out, cv::IImageBatch &kernel,
                               cv::ITensor &kernelAnchor, NVCVBorderType borderMode)
{
    cv::detail::CheckThrow(nvcvopConv2DVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), kernel.handle(),
                                                      kernelAnchor.handle(), borderMode));
}

inline NVCVOperatorHandle Conv2D::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_CONV2D_HPP
