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

#include "OpGammaContrast.hpp"

#include <nvcv/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

GammaContrast::GammaContrast(const int32_t maxVarShapeBatchSize, const int32_t maxVarShapeChannelCount)
{
    m_legacyOpVarShape
        = std::make_unique<leg::cuda_op::GammaContrastVarShape>(maxVarShapeBatchSize, maxVarShapeChannelCount);
}

void GammaContrast::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                               const cv::ITensor &gamma) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input must be device-acessible, varshape pitch-linear image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedCuda *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be device-acessible, varshape pitch-linear image batch");
    }

    auto *gammaData = dynamic_cast<const cv::ITensorDataStridedCuda *>(gamma.exportData());
    if (gammaData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Gamma must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, *gammaData, stream));
}

} // namespace nv::cvop::priv
