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

#include "OpPadAndStack.hpp"

#include "Exception.hpp"

#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

PadAndStack::PadAndStack()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<leg::cuda_op::PadAndStack>(maxIn, maxOut);
}

void PadAndStack::operator()(cudaStream_t stream, cv::IImageBatch &in, cv::ITensor &out, cv::ITensor &left,
                             cv::ITensor &top, const NVCVBorderType borderMode, const float borderValue) const
{
    auto *inData = dynamic_cast<const cv::IImageBatchVarShapeDataDevicePitch *>(in.exportData(stream));
    if (inData == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-accessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-accessible, pitch-linear tensor");
    }

    auto *leftData = dynamic_cast<const cv::ITensorDataPitchDevice *>(left.exportData());
    if (outData == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Left must be device-accessible, pitch-linear tensor");
    }

    auto *topData = dynamic_cast<const cv::ITensorDataPitchDevice *>(top.exportData());
    if (outData == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Top must be device-accessible, pitch-linear tensor");
    }

    leg::helpers::CheckOpErrThrow(
        m_legacyOp->infer(*inData, *outData, *topData, *leftData, borderMode, borderValue, stream));
}

nv::cv::priv::Version PadAndStack::doGetVersion() const
{
    //todo need to have a version decoupled from NVCV
    return nv::cv::priv::CURRENT_VERSION;
}

} // namespace nv::cvop::priv
