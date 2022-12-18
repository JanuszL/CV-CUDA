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

#include "legacy/CvCudaLegacy.h"
#include "legacy/CvCudaLegacyHelpers.hpp"

#include <nvcv/Exception.hpp>
#include <util/CheckError.hpp>

namespace nvcvop::priv {

namespace legacy = nvcv::legacy::cuda_op;

BilateralFilter::BilateralFilter()
{
    legacy::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<legacy::BilateralFilter>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<legacy::BilateralFilterVarShape>(maxIn, maxOut);
}

void BilateralFilter::operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out, int diameter,
                                 float sigmaColor, float sigmaSpace, NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(in.exportData());
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(out.exportData());
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, diameter, sigmaColor, sigmaSpace, borderMode, stream));
}

void BilateralFilter::operator()(cudaStream_t stream, const nvcv::IImageBatchVarShape &in,
                                 const nvcv::IImageBatchVarShape &out, const nvcv::ITensor &diameter,
                                 const nvcv::ITensor &sigmaColor, const nvcv::ITensor &sigmaSpace,
                                 NVCVBorderType borderMode) const
{
    auto *inData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be device-accessible, varshape image batch");
    }

    auto *outData = dynamic_cast<const nvcv::IImageBatchVarShapeDataStridedCuda *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be device-accessible,  varshape image batch");
    }

    auto *diameterData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(diameter.exportData());
    if (diameterData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Diameter must be device-accessible, pitch-linear tensor");
    }

    auto *sigmaColorData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(sigmaColor.exportData());
    if (sigmaColorData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigmaColor must be device-accessible, pitch-linear tensor");
    }

    auto *sigmaSpaceData = dynamic_cast<const nvcv::ITensorDataStridedCuda *>(sigmaSpace.exportData());
    if (sigmaSpaceData == nullptr)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "sigmaSpace must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(m_legacyOpVarShape->infer(*inData, *outData, *diameterData, *sigmaColorData, *sigmaSpaceData,
                                               borderMode, stream));
}

} // namespace nvcvop::priv
