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
 * @file OpGammaContrast.hpp
 *
 * @brief Defines the public C++ Class for the Gamma Contrast operation.
 * @defgroup NVCV_CPP_ALGORITHM_GAMMA Contrast
 * @{
 */

#ifndef NVCV_OP_GAMMA_HPP
#define NVCV_OP_GAMMA_HPP

#include "IOperator.hpp"
#include "OpGammaContrast.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class GammaContrast final : public IOperator
{
public:
    explicit GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount);

    ~GammaContrast();

    void operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out, cv::ITensor &gamma);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline GammaContrast::GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount)
{
    cv::detail::CheckThrow(nvcvopGammaContrastCreate(&m_handle, maxVarShapeBatchSize, maxVarShapeChannelCount));
    assert(m_handle);
}

inline GammaContrast::~GammaContrast()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void GammaContrast::operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out,
                                      cv::ITensor &gamma)
{
    cv::detail::CheckThrow(
        nvcvopGammaContrastVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), gamma.handle()));
}

inline NVCVOperatorHandle GammaContrast::handle() const noexcept
{
    return m_handle;
}

}}     // namespace nv::cvop
#endif // NVCV_OP_GAMMA_HPP
