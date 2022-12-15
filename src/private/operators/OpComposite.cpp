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

#include "OpComposite.hpp"

#include <nvcv/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Composite::Composite()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::Composite>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::CompositeVarShape>(maxIn, maxOut);
}

void Composite::operator()(cudaStream_t stream, const cv::ITensor &foreground, const cv::ITensor &background,
                           const cv::ITensor &fgMask, const cv::ITensor &output) const
{
    auto *foregroundData = dynamic_cast<const cv::ITensorDataStridedDevice *>(foreground.exportData());
    if (foregroundData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input foreground must be device-acessible, pitch-linear tensor");
    }

    auto *backgroundData = dynamic_cast<const cv::ITensorDataStridedDevice *>(background.exportData());
    if (backgroundData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input background must be device-acessible, pitch-linear tensor");
    }

    auto *fgMaskData = dynamic_cast<const cv::ITensorDataStridedDevice *>(fgMask.exportData());
    if (fgMaskData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input fgMask must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataStridedDevice *>(output.exportData());
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Output must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*foregroundData, *backgroundData, *fgMaskData, *outData, stream));
}

void Composite::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &foreground,
                           const cv::IImageBatchVarShape &background, const cv::IImageBatchVarShape &fgMask,
                           const cv::IImageBatchVarShape &output) const
{
    auto *foregroundData
        = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(foreground.exportData(stream));
    if (foregroundData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input foreground must be device-acessible, varshape image batch");
    }

    auto *backgroundData
        = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(background.exportData(stream));
    if (backgroundData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input background must be device-acessible, varshape image batch");
    }

    auto *fgMaskData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(fgMask.exportData(stream));
    if (fgMaskData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input fgMask must be device-acessible, varshape image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(output.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be device-acessible, varshape image batch");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*foregroundData, *backgroundData, *fgMaskData, *outData, stream));
}

} // namespace nv::cvop::priv
