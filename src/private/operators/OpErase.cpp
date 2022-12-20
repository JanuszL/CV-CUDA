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

#include "OpErase.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Erase::Erase(int num_erasing_area)
{
    leg::cuda_op::DataShape maxIn, maxOut;
    // maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::Erase>(maxIn, maxOut, num_erasing_area);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::EraseVarShape>(maxIn, maxOut, num_erasing_area);
}

void Erase::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::ITensor &anchor,
                       cv::ITensor &erasing, cv::ITensor &values, cv::ITensor &imgIdx, bool random,
                       unsigned int seed) const
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

    auto *anchorData = dynamic_cast<const cv::ITensorDataStridedCuda *>(anchor.exportData());
    if (anchorData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "anchor must be cuda-accessible, pitch-linear tensor");
    }

    auto *erasingData = dynamic_cast<const cv::ITensorDataStridedCuda *>(erasing.exportData());
    if (erasingData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "erasing must be cuda-accessible, pitch-linear tensor");
    }

    auto *valuesData = dynamic_cast<const cv::ITensorDataStridedCuda *>(values.exportData());
    if (valuesData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "values must be cuda-accessible, pitch-linear tensor");
    }

    auto *imgIdxData = dynamic_cast<const cv::ITensorDataStridedCuda *>(imgIdx.exportData());
    if (imgIdxData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "imgIdx must be cuda-accessible, pitch-linear tensor");
    }

    bool inplace = (in.handle() == out.handle());
    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, *anchorData, *erasingData, *valuesData, *imgIdxData, random,
                                       seed, inplace, stream));
}

void Erase::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                       cv::ITensor &anchor, cv::ITensor &erasing, cv::ITensor &values, cv::ITensor &imgIdx, bool random,
                       unsigned int seed) const
{
    auto *anchorData = dynamic_cast<const cv::ITensorDataStridedCuda *>(anchor.exportData());
    if (anchorData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "anchor must be cuda-accessible, pitch-linear tensor");
    }

    auto *erasingData = dynamic_cast<const cv::ITensorDataStridedCuda *>(erasing.exportData());
    if (erasingData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "erasing must be cuda-accessible, pitch-linear tensor");
    }

    auto *valuesData = dynamic_cast<const cv::ITensorDataStridedCuda *>(values.exportData());
    if (valuesData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "values must be cuda-accessible, pitch-linear tensor");
    }

    auto *imgIdxData = dynamic_cast<const cv::ITensorDataStridedCuda *>(imgIdx.exportData());
    if (imgIdxData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "imgIdx must be cuda-accessible, pitch-linear tensor");
    }

    bool inplace = (in.handle() == out.handle());
    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(in, out, *anchorData, *erasingData, *valuesData, *imgIdxData, random,
                                               seed, inplace, stream));
}

} // namespace nv::cvop::priv
