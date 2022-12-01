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

#include "OpCopyMakeBorder.hpp"

#include <nvcv/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

CopyMakeBorder::CopyMakeBorder()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::CopyMakeBorder>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::CopyMakeBorderVarShape>(maxIn, maxOut);
}

void CopyMakeBorder::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, const int top,
                                const int left, const NVCVBorderType borderMode, const float4 borderValue) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataPitchDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Input must be device-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be device-accessible, pitch-linear tensor");
    }

    leg::helpers::CheckOpErrThrow(m_legacyOp->infer(*inData, *outData, top, left, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const cv::IImageBatch &in, const cv::ITensor &out,
                                const cv::ITensor &top, const cv::ITensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "Output must be device-accessible, pitch-linear tensor");
    }

    auto *topData = dynamic_cast<const cv::ITensorDataPitchDevice *>(top.exportData());
    if (topData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Top must be device-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const cv::ITensorDataPitchDevice *>(left.exportData());
    if (leftData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Left must be device-accessible, pitch-linear tensor");
    }

    leg::helpers::CheckOpErrThrow(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const cv::IImageBatch &in, const cv::IImageBatch &out,
                                const cv::ITensor &top, const cv::ITensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto *topData = dynamic_cast<const cv::ITensorDataPitchDevice *>(top.exportData());
    if (topData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Top must be device-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const cv::ITensorDataPitchDevice *>(left.exportData());
    if (leftData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Left must be device-accessible, pitch-linear tensor");
    }

    leg::helpers::CheckOpErrThrow(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

} // namespace nv::cvop::priv
