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

#include "OpCopyMakeBorder.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

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

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, top, left, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const cv::IImageBatch &in, const cv::ITensor &out,
                                const cv::ITensor &top, const cv::ITensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataStridedDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-accessible, pitch-linear tensor");
    }

    auto *topData = dynamic_cast<const cv::ITensorDataStridedDevice *>(top.exportData());
    if (topData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Top must be device-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const cv::ITensorDataStridedDevice *>(left.exportData());
    if (leftData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Left must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

void CopyMakeBorder::operator()(cudaStream_t stream, const cv::IImageBatch &in, const cv::IImageBatch &out,
                                const cv::ITensor &top, const cv::ITensor &left, const NVCVBorderType borderMode,
                                const float4 borderValue) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be varshape image batch");
    }

    auto *outData = dynamic_cast<const cv::IImageBatchVarShapeDataStridedDevice *>(out.exportData(stream));
    if (outData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be varshape image batch");
    }

    auto *topData = dynamic_cast<const cv::ITensorDataStridedDevice *>(top.exportData());
    if (topData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Top must be device-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const cv::ITensorDataStridedDevice *>(left.exportData());
    if (leftData == nullptr)
    {
        throw cv::priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Left must be device-accessible, pitch-linear tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

} // namespace nv::cvop::priv
