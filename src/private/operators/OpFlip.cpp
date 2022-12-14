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
    auto *input = dynamic_cast<const cv::ITensorDataPitchDevice *>(in.exportData());
    if (input == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Input must be device-accessible, pitch-linear tensor");
    }

    auto *output = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (output == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*input, *output, flipCode, stream));
}

void Flip::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                      const cv::ITensor &flipCode) const
{
    auto *input = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(in.exportData(stream));
    if (input == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Input must be device-acessible, varshape pitch-linear "
                            "image batch");
    }

    auto *output = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(out.exportData(stream));
    if (output == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be device-acessible, varshape pitch-linear"
                            " image batch");
    }

    auto *flip_code = dynamic_cast<const cv::ITensorDataPitchDevice *>(flipCode.exportData());
    if (flip_code == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Flip Code must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*input, *output, *flip_code, stream));
}

} // namespace nv::cvop::priv
