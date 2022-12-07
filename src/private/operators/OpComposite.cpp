/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
    m_legacyOp = std::make_unique<leg::cuda_op::Composite>(maxIn, maxOut);
}

void Composite::operator()(cudaStream_t stream, const cv::ITensor &foreground, const cv::ITensor &background,
                           const cv::ITensor &mat, const cv::ITensor &output) const
{
    auto *foregroundData = dynamic_cast<const cv::ITensorDataPitchDevice *>(foreground.exportData());
    if (foregroundData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input foreground must be device-acessible, pitch-linear tensor");
    }

    auto *backgroundData = dynamic_cast<const cv::ITensorDataPitchDevice *>(background.exportData());
    if (backgroundData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input background must be device-acessible, pitch-linear tensor");
    }

    auto *matData = dynamic_cast<const cv::ITensorDataPitchDevice *>(mat.exportData());
    if (matData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input mat must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(output.exportData());
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Output must be device-acessible, pitch-linear tensor");
    }
    
    NVCV_CHECK_THROW(
    m_legacyOp->infer(*foregroundData, *backgroundData, *matData, *outData, stream));
}

} // namespace nv::cvop::priv
