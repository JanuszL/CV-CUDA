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

#include "OpNormalize.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Normalize::Normalize()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::Normalize>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::NormalizeVarShape>(maxIn, maxOut);
}

void Normalize::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &base,
                           const cv::ITensor &scale, cv::ITensor &out, const float global_scale, const float shift,
                           const float epsilon, const uint32_t flags) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataStridedDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-acessible, pitch-linear tensor");
    }

    auto *baseData = dynamic_cast<const cv::ITensorDataStridedDevice *>(base.exportData());
    if (baseData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Input base must be device-acessible, pitch-linear tensor");
    }

    auto *scaleData = dynamic_cast<const cv::ITensorDataStridedDevice *>(scale.exportData());
    if (scaleData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Input scale must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataStridedDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOp->infer(*inData, *baseData, *scaleData, *outData, global_scale, shift, epsilon, flags, stream));
}

void Normalize::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::ITensor &base,
                           const cv::ITensor &scale, cv::IImageBatchVarShape &out, const float global_scale,
                           const float shift, const float epsilon, const uint32_t flags) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Input must be device-acessible, varshape pitch-linear image batch");
    }

    auto *baseData = dynamic_cast<const cv::ITensorDataStridedDevice *>(base.exportData());
    if (baseData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Input base must be device-acessible, pitch-linear tensor");
    }

    auto *scaleData = dynamic_cast<const cv::ITensorDataStridedDevice *>(scale.exportData());
    if (scaleData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Input scale must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Output must be device-acessible, varshape pitch-linear image batch");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *baseData, *scaleData, *outData, global_scale, shift, epsilon,
                                               flags, stream));
}

} // namespace nv::cvop::priv
