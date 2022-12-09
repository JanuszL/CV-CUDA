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

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Morphology::Morphology(const int32_t maxVarShapeBatchSize)
{
    leg::cuda_op::DataShape maxIn, maxOut;
    //maxIn maxOut not used by ctor
    m_legacyOp         = std::make_unique<leg::cuda_op::Morphology>(maxIn, maxOut);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::MorphologyVarShape>(maxVarShapeBatchSize);
}

void Morphology::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out,
                            NVCVMorphologyType morph_type, cv::Size2D mask_size, int2 anchor, int32_t iteration,
                            const NVCVBorderType borderMode) const
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

    NVCV_CHECK_THROW(
        m_legacyOp->infer(*inData, *outData, morph_type, mask_size, anchor, iteration, borderMode, stream));
}

void Morphology::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                            NVCVMorphologyType morph_type, cv::ITensor &masks, cv::ITensor &anchors, int32_t iteration,
                            NVCVBorderType borderMode) const
{
    auto *masksData = dynamic_cast<const cv::ITensorDataPitchDevice *>(masks.exportData());
    if (masksData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "masksData must be a tensor");
    }

    auto *anchorsData = dynamic_cast<const cv::ITensorDataPitchDevice *>(anchors.exportData());
    if (anchorsData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "anchors must be a tensor");
    }

    NVCV_CHECK_THROW(
        m_legacyOpVarShape->infer(in, out, morph_type, *masksData, *anchorsData, iteration, borderMode, stream));
}

} // namespace nv::cvop::priv
