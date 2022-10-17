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

#include "OpReformat.hpp"

#include "Exception.hpp"

#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Reformat::Reformat()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn/maxOut not used by op.
    m_legacyOp = std::make_unique<leg::cuda_op::Reformat>(maxIn, maxOut);
}

void Reformat::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataPitchDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Output must be device-acessible, pitch-linear tensor");
    }

    leg::helpers::CheckOpErrThrow(m_legacyOp->infer(*inData, *outData, stream));
}

cv::priv::Version Reformat::doGetVersion() const
{
    //todo need to have a version decoupled from NVCV
    return cv::priv::CURRENT_VERSION;
}

} // namespace nv::cvop::priv
