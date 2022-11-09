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
 * @file OpConv2D.hpp
 *
 * @brief Defines the private C++ Class for the 2D convolution operation.
 */

#ifndef NVCV_OP_PRIV_CONV2D_HPP
#define NVCV_OP_PRIV_CONV2D_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>

#include <memory>

namespace nv::cvop::priv {

class Conv2D final : public IOperator
{
public:
    explicit Conv2D();

    void operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                    const cv::IImageBatchVarShape &kernel, const cv::ITensor &kernelAnchor,
                    NVCVBorderType borderMode) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::Conv2DVarShape> m_legacyOpVarShape;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_CONV2D_HPP
