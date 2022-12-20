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

#include "OpAverageBlur.hpp"

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace legacy = cv::legacy::cuda_op;

AverageBlur::AverageBlur(cv::Size2D maxKernelSize, int maxBatchSize)
{
    legacy::DataShape maxIn, maxOut; //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::AverageBlur>(maxIn, maxOut, maxKernelSize);
    m_legacyOpVarShape = std::make_unique<legacy::AverageBlurVarShape>(maxIn, maxOut, maxKernelSize, maxBatchSize);
}

void AverageBlur::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::Size2D kernelSize,
                             int2 kernelAnchor, NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataStridedCuda *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, kernelSize, kernelAnchor, borderMode, stream));
}

void AverageBlur::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                             const cv::ITensor &kernelSize, const cv::ITensor &kernelAnchor,
                             NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedCuda *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be cuda-accessible, varshape pitch-linear image batch");
    }

    auto *kernelSizeData = dynamic_cast<const cv::ITensorDataStridedCuda *>(kernelSize.exportData());
    if (kernelSizeData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Kernel size must be cuda-accessible, pitch-linear tensor");
    }

    auto *kernelAnchorData = dynamic_cast<const cv::ITensorDataStridedCuda *>(kernelAnchor.exportData());
    if (kernelAnchorData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Kernel anchor must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *kernelSizeData, *kernelAnchorData, borderMode, stream));
}

} // namespace nv::cvop::priv
