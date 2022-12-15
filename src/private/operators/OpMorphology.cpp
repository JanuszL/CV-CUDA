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

#include "OpMorphology.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Morphology::Morphology(const int32_t maxVarShapeBatchSize)
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn maxOut not used by ctor
    m_legacyOp         = std::make_unique<leg::cuda_op::Morphology>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::MorphologyVarShape>(maxVarShapeBatchSize);
}

void Morphology::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out,
                            NVCVMorphologyType morph_type, cv::Size2D mask_size, int2 anchor, int32_t iteration,
                            const NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataStridedCuda *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOp->infer(*inData, *outData, morph_type, mask_size, anchor, iteration, borderMode, stream));
}

void Morphology::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                            NVCVMorphologyType morph_type, cv::ITensor &masks, cv::ITensor &anchors, int32_t iteration,
                            NVCVBorderType borderMode) const
{
    auto *masksData = dynamic_cast<const cv::ITensorDataStridedCuda *>(masks.exportData());
    if (masksData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "masksData must be a tensor");
    }

    auto *anchorsData = dynamic_cast<const cv::ITensorDataStridedCuda *>(anchors.exportData());
    if (anchorsData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "anchors must be a tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(in, out, morph_type, *masksData, *anchorsData, iteration, borderMode, stream));
}

} // namespace nv::cvop::priv
