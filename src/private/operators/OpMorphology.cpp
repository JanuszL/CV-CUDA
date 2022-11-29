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

#include "OpMorphology.hpp"

#include <nvcv/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Morphology::Morphology()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn maxOut not used by ctor
    m_legacyOp = std::make_unique<leg::cuda_op::Morphology>(maxIn, maxOut);
}

void Morphology::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out,
                            NVCVMorphologyType morph_type, cv::Size2D mask_size, int2 anchor, int iteration,
                            const NVCVBorderType borderMode) const
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

    leg::helpers::CheckOpErrThrow(
        m_legacyOp->infer(*inData, *outData, morph_type, mask_size, anchor, iteration, borderMode, stream));
}

} // namespace nv::cvop::priv
