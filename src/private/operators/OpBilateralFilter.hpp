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
 * @file OpBilateralFilter.hpp
 *
 * @brief Defines the private C++ Class for the Bilateral Filter operation.
 */

#ifndef NVCV_OP_PRIV_BILATERAL_FILTER_HPP
#define NVCV_OP_PRIV_BILATERAL_FILTER_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/alloc/Requirements.hpp>
#include <private/core/Exception.hpp>
#include <private/core/ICoreObject.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/Version.hpp>
#include <private/legacy/CvCudaLegacy.h>

#include <memory>

// Use the public nvcv API
namespace nv::cvop::priv {

class BilateralFilter final : public OperatorBase
{
public:
    explicit BilateralFilter();

    void operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, int diameter, float sigmaColor,
                    float sigmaSpace, NVCVBorderType borderMode) const;

    void operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                    const cv::ITensor &diameter, const cv::ITensor &sigmaColor, const cv::ITensor &sigmaSpace,
                    NVCVBorderType borderMode) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::BilateralFilter>         m_legacyOp;
    std::unique_ptr<cv::legacy::cuda_op::BilateralFilterVarShape> m_legacyOpVarShape;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_BILATERAL_FILTER_HPP
