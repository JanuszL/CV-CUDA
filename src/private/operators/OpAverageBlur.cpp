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

#include "OpAverageBlur.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

AverageBlur::AverageBlur(cv::Size2D maxKernelSize, int maxBatchSize)
{
    leg::cuda_op::DataShape maxIn, maxOut; //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<leg::cuda_op::AverageBlur>(maxIn, maxOut, maxKernelSize);
    m_legacyOpVarShape
        = std::make_unique<leg::cuda_op::AverageBlurVarShape>(maxIn, maxOut, maxKernelSize, maxBatchSize);
}

void AverageBlur::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::Size2D kernelSize,
                             int2 kernelAnchor, NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataPitchDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, kernelSize, kernelAnchor, borderMode, stream));
}

void AverageBlur::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                             const cv::ITensor &kernelSize, const cv::ITensor &kernelAnchor,
                             NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Input must be device-acessible, varshape pitch-linear image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Output must be device-acessible, varshape pitch-linear image batch");
    }

    auto *kernelSizeData = dynamic_cast<const cv::ITensorDataPitchDevice *>(kernelSize.exportData());
    if (kernelSizeData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Kernel size must be device-acessible, pitch-linear tensor");
    }

    auto *kernelAnchorData = dynamic_cast<const cv::ITensorDataPitchDevice *>(kernelAnchor.exportData());
    if (kernelAnchorData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Kernel anchor must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *kernelSizeData, *kernelAnchorData, borderMode, stream));
}

} // namespace nv::cvop::priv
