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

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

AverageBlur::AverageBlur(cv::Size2D maxKernelSize, int maxBatchSize)
{
    leg::cuda_op::DataShape maxIn, maxOut; //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<leg::cuda_op::AverageBlur>(maxIn, maxOut, maxKernelSize);
    m_legacyOpVarShape
        = std::make_unique<leg::cuda_op::AverageBlurVarShape>(maxIn, maxOut, maxKernelSize, maxBatchSize);
}

void AverageBlur::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::Size2D kernelSize,
                             int2 kernelAnchor, NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataPitchDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, kernelSize, kernelAnchor, borderMode, stream));
}

void AverageBlur::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                             const cv::ITensor &kernelSize, const cv::ITensor &kernelAnchor,
                             NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Input must be device-acessible, varshape pitch-linear image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Output must be device-acessible, varshape pitch-linear image batch");
    }

    auto *kernelSizeData = dynamic_cast<const cv::ITensorDataPitchDevice *>(kernelSize.exportData());
    if (kernelSizeData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Kernel size must be device-acessible, pitch-linear tensor");
    }

    auto *kernelAnchorData = dynamic_cast<const cv::ITensorDataPitchDevice *>(kernelAnchor.exportData());
    if (kernelAnchorData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Kernel anchor must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *kernelSizeData, *kernelAnchorData, borderMode, stream));
}

} // namespace nv::cvop::priv
