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
 * @file OpComposite.hpp
 *
 * @brief Defines the public C++ Class for the Composite operation.
 * @defgroup NVCV_CPP_ALGORITHM_COMPOSITE Composite
 * @{
 */

#ifndef NVCV_OP_COMPOSITE_HPP
#define NVCV_OP_COMPOSITE_HPP

#include "IOperator.hpp"
#include "OpComposite.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Composite final : public IOperator
{
public:
    explicit Composite();

    ~Composite();

    void operator()(cudaStream_t stream, cv::ITensor &foreground, cv::ITensor &background, cv::ITensor &fgMask,
                    cv::ITensor &output);

    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &foreground, cv::IImageBatchVarShape &background,
                    cv::IImageBatchVarShape &fgMask, cv::IImageBatchVarShape &output);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Composite::Composite()
{
    cv::detail::CheckThrow(nvcvopCompositeCreate(&m_handle));
    assert(m_handle);
}

inline Composite::~Composite()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Composite::operator()(cudaStream_t stream, cv::ITensor &foreground, cv::ITensor &background,
                                  cv::ITensor &fgMask, cv::ITensor &output)
{
    cv::detail::CheckThrow(nvcvopCompositeSubmit(m_handle, stream, foreground.handle(), background.handle(),
                                                 fgMask.handle(), output.handle()));
}

inline void Composite::operator()(cudaStream_t stream, cv::IImageBatchVarShape &foreground,
                                  cv::IImageBatchVarShape &background, cv::IImageBatchVarShape &fgMask,
                                  cv::IImageBatchVarShape &output)
{
    cv::detail::CheckThrow(nvcvopCompositeVarShapeSubmit(m_handle, stream, foreground.handle(), background.handle(),
                                                         fgMask.handle(), output.handle()));
}

inline NVCVOperatorHandle Composite::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_COMPOSITE_HPP
