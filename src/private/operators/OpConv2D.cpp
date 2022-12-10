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

#include "OpConv2D.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Conv2D::Conv2D()
{
    leg::cuda_op::DataShape maxIn, maxOut; //maxIn/maxOut not used by op.
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::Conv2DVarShape>(maxIn, maxOut);
}

void Conv2D::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                        const cv::IImageBatchVarShape &kernel, const cv::ITensor &kernelAnchor,
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

    auto *kernelData = dynamic_cast<const cv::IImageBatchVarShapeDataPitchDevice *>(kernel.exportData(stream));
    if (kernelData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Kernel must be device-acessible, varshape pitch-linear image batch");
    }

    auto *kernelAnchorData = dynamic_cast<const cv::ITensorDataPitchDevice *>(kernelAnchor.exportData());
    if (kernelAnchorData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Kernel anchor must be device-acessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, *kernelData, *kernelAnchorData, borderMode, stream));
}

} // namespace nv::cvop::priv
