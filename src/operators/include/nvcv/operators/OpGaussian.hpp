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
 * @file OpGaussian.hpp
 *
 * @brief Defines the public C++ class for the Gaussian operation.
 * @defgroup NVCV_CPP_ALGORITHM_GAUSSIAN Gaussian
 * @{
 */

#ifndef NVCV_OP_GAUSSIAN_HPP
#define NVCV_OP_GAUSSIAN_HPP

#include "IOperator.hpp"
#include "OpGaussian.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Gaussian final : public IOperator
{
public:
    explicit Gaussian(cv::Size2D maxKernelSize, int32_t maxVarShapeBatchSize);

    ~Gaussian();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::Size2D kernelSize, double2 sigma,
                    NVCVBorderType borderMode);

    void operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out, cv::ITensor &kernelSize,
                    cv::ITensor &sigma, NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Gaussian::Gaussian(cv::Size2D maxKernelSize, int32_t maxVarShapeBatchSize)
{
    cv::detail::CheckThrow(nvcvopGaussianCreate(&m_handle, maxKernelSize.w, maxKernelSize.h, maxVarShapeBatchSize));
    assert(m_handle);
}

inline Gaussian::~Gaussian()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Gaussian::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::Size2D kernelSize,
                                 double2 sigma, NVCVBorderType borderMode)
{
    cv::detail::CheckThrow(nvcvopGaussianSubmit(m_handle, stream, in.handle(), out.handle(), kernelSize.w, kernelSize.h,
                                                sigma.x, sigma.y, borderMode));
}

inline void Gaussian::operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out,
                                 cv::ITensor &kernelSize, cv::ITensor &sigma, NVCVBorderType borderMode)
{
    cv::detail::CheckThrow(nvcvopGaussianVarShapeSubmit(m_handle, stream, in.handle(), out.handle(),
                                                        kernelSize.handle(), sigma.handle(), borderMode));
}

inline NVCVOperatorHandle Gaussian::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_GAUSSIAN_HPP
