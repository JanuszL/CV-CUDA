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

#include "OpPadAndStack.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

PadAndStack::PadAndStack()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<leg::cuda_op::PadAndStack>(maxIn, maxOut);
}

void PadAndStack::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::ITensor &out, cv::ITensor &top,
                             cv::ITensor &left, const NVCVBorderType borderMode, const float borderValue) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto *topData = dynamic_cast<const cv::ITensorDataStridedCuda *>(top.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Top must be cuda-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const cv::ITensorDataStridedCuda *>(left.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Left must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

} // namespace nv::cvop::priv
