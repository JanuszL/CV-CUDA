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
 * @file OpAverageBlur.hpp
 *
 * @brief Defines the public C++ class for the AverageBlur operation.
 * @defgroup NVCV_CPP_ALGORITHM_AVERAGEBLUR Average Blur
 * @{
 */

#ifndef NVCV_OP_AVERAGEBLUR_HPP
#define NVCV_OP_AVERAGEBLUR_HPP

#include "IOperator.hpp"
#include "OpAverageBlur.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class AverageBlur final : public IOperator
{
public:
    explicit AverageBlur(cv::Size2D maxKernelSize, int32_t maxVarShapeBatchSize);

    ~AverageBlur();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::Size2D kernelSize, int2 kernelAnchor,
                    NVCVBorderType borderMode);

    void operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out, cv::ITensor &kernelSize,
                    cv::ITensor &kernelAnchor, NVCVBorderType borderMode);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline AverageBlur::AverageBlur(cv::Size2D maxKernelSize, int32_t maxVarShapeBatchSize)
{
    cv::detail::CheckThrow(nvcvopAverageBlurCreate(&m_handle, maxKernelSize.w, maxKernelSize.h, maxVarShapeBatchSize));
    assert(m_handle);
}

inline AverageBlur::~AverageBlur()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void AverageBlur::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::Size2D kernelSize,
                                    int2 kernelAnchor, NVCVBorderType borderMode)
{
    cv::detail::CheckThrow(nvcvopAverageBlurSubmit(m_handle, stream, in.handle(), out.handle(), kernelSize.w,
                                                   kernelSize.h, kernelAnchor.x, kernelAnchor.y, borderMode));
}

inline void AverageBlur::operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out,
                                    cv::ITensor &kernelSize, cv::ITensor &kernelAnchor, NVCVBorderType borderMode)
{
    cv::detail::CheckThrow(nvcvopAverageBlurVarShapeSubmit(m_handle, stream, in.handle(), out.handle(),
                                                           kernelSize.handle(), kernelAnchor.handle(), borderMode));
}

inline NVCVOperatorHandle AverageBlur::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_AVERAGEBLUR_HPP
