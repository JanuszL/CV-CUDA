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

#include "OpResize.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Resize::Resize()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    // maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::Resize>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::ResizeVarShape>(maxIn, maxOut);
}

void Resize::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out,
                        const NVCVInterpolationType interpolation) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataPitchDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, interpolation, stream));
}

void Resize::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                        const NVCVInterpolationType interpolation) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, interpolation, stream));
}

} // namespace nv::cvop::priv
