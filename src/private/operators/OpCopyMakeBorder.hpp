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
 * @file OpCopyMakeBorder.hpp
 *
 * @brief Defines the private C++ class for the CopyMakeBorder operation.
 */

#ifndef NVCV_OP_PRIV_COPYMAKEBORDER_HPP
#define NVCV_OP_PRIV_COPYMAKEBORDER_HPP

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

namespace nv::cvop::priv {

class CopyMakeBorder final : public OperatorBase
{
public:
    explicit CopyMakeBorder();

    void operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, const int top, const int left,
                    const NVCVBorderType borderMode, const float4 borderValue) const;
    void operator()(cudaStream_t stream, const cv::IImageBatch &in, const cv::IImageBatch &out, const cv::ITensor &top,
                    const cv::ITensor &left, const NVCVBorderType borderMode, const float4 borderValue) const;
    void operator()(cudaStream_t stream, const cv::IImageBatch &in, const cv::ITensor &out, const cv::ITensor &top,
                    const cv::ITensor &left, const NVCVBorderType borderMode, const float4 borderValue) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::CopyMakeBorder>         m_legacyOp;
    std::unique_ptr<cv::legacy::cuda_op::CopyMakeBorderVarShape> m_legacyOpVarShape;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_PADANDSTACK_HPP
