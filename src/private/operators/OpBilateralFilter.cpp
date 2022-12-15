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

#include "OpBilateralFilter.hpp"

#include <nvcv/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

BilateralFilter::BilateralFilter()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::BilateralFilter>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::BilateralFilterVarShape>(maxIn, maxOut);
}

void BilateralFilter::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, int diameter,
                                 float sigmaColor, float sigmaSpace, NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataStridedDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataStridedDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, diameter, sigmaColor, sigmaSpace, borderMode, stream));
}

void BilateralFilter::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in,
                                 const cv::IImageBatchVarShape &out, const cv::ITensor &diameter,
                                 const cv::ITensor &sigmaColor, const cv::ITensor &sigmaSpace,
                                 NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-accessible, varshape image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Output must be device-accessible,  varshape image batch");
    }

    auto *diameterData = dynamic_cast<const cv::ITensorDataStridedDevice *>(diameter.exportData());
    if (diameterData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Diameter must be device-accessible, pitch-linear tensor");
    }

    auto *sigmaColorData = dynamic_cast<const cv::ITensorDataStridedDevice *>(sigmaColor.exportData());
    if (sigmaColorData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "sigmaColor must be device-accessible, pitch-linear tensor");
    }

    auto *sigmaSpaceData = dynamic_cast<const cv::ITensorDataStridedDevice *>(sigmaSpace.exportData());
    if (sigmaSpaceData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "sigmaSpace must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, *diameterData, *sigmaColorData, *sigmaSpaceData,
                                               borderMode, stream));
}

} // namespace nv::cvop::priv
