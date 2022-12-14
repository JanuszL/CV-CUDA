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

#include "OpCvtColor.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

CvtColor::CvtColor()
{
    leg::cuda_op::DataShape maxIn, maxOut; //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::CvtColor>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::CvtColorVarShape>(maxIn, maxOut);
}

void CvtColor::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out,
                          NVCVColorConversionCode code) const
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

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, code, stream));
}

void CvtColor::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                          NVCVColorConversionCode code) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input must be device-acessible, varshape pitch-linear image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be device-acessible, varshape pitch-linear image batch");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, code, stream));
}

} // namespace nv::cvop::priv
