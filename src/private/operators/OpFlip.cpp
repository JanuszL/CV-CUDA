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

#include "OpFlip.hpp"

#include <nvcv/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Flip::Flip(int32_t maxBatchSize)
{
    leg::cuda_op::DataShape maxIn, maxOut; //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::Flip>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::FlipOrCopyVarShape>(maxIn, maxOut);
}

void Flip::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, int32_t flipCode) const
{
    auto *input = dynamic_cast<const cv::ITensorDataStridedCuda *>(in.exportData());
    if (input == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto *output = dynamic_cast<const cv::ITensorDataStridedCuda *>(out.exportData());
    if (output == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*input, *output, flipCode, stream));
}

void Flip::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                      const cv::ITensor &flipCode) const
{
    auto *input = dynamic_cast<const cv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (input == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input must be cuda-accessible, varshape pitch-linear "
                            "image batch");
    }

    auto *output = dynamic_cast<const cv::IImageBatchVarShapeDataStridedCuda *>(out.exportData(stream));
    if (output == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be cuda-accessible, varshape pitch-linear"
                            " image batch");
    }

    auto *flip_code = dynamic_cast<const cv::ITensorDataStridedCuda *>(flipCode.exportData());
    if (flip_code == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Flip Code must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*input, *output, *flip_code, stream));
}

} // namespace nv::cvop::priv
