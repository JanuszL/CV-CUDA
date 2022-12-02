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

#include "OpWarpPerspective.hpp"

#include "nvcv/Exception.hpp"

#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

WarpPerspective::WarpPerspective(const int32_t maxVarShapeBatchSize)
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::WarpPerspective>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::WarpPerspectiveVarShape>(maxVarShapeBatchSize);
}

void WarpPerspective::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out,
                                 const NVCVPerspectiveTransform transMatrix, const int32_t flags,
                                 const NVCVBorderType borderMode, const float4 borderValue) const
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

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, transMatrix, flags, borderMode, borderValue, stream));
}

void WarpPerspective::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in,
                                 const cv::IImageBatchVarShape &out, const cv::ITensor &transMatrix,
                                 const int32_t flags, const NVCVBorderType borderMode, const float4 borderValue) const
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

    auto *transMatrixData = dynamic_cast<const cv::ITensorDataPitchDevice *>(transMatrix.exportData());
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "transformation matrix must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *transMatrixData, flags, borderMode, borderValue, stream));
}

} // namespace nv::cvop::priv
