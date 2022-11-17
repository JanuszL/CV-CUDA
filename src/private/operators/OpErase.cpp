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

#include "OpErase.hpp"

#include "nvcv/Exception.hpp"

#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

Erase::Erase()
{
    leg::cuda_op::DataShape maxIn, maxOut;
    // maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::Erase>(maxIn, maxOut);
}

void Erase::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::ITensor &anchor_x, cv::ITensor &anchor_y, 
                    cv::ITensor &erasing_w, cv::ITensor &erasing_h, cv::ITensor &erasing_c, cv::ITensor &values, cv::ITensor &imgIdx, 
                    int max_eh, int max_ew, bool random, unsigned int seed, bool inplace) const
{
    auto *inData = dynamic_cast<const cv::ITensorDataPitchDevice *>(in.exportData());
    if (inData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Input must be device-acessible, pitch-linear tensor");
    }

    auto *outData = dynamic_cast<const cv::ITensorDataPitchDevice *>(out.exportData());
    if (outData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Output must be device-acessible, pitch-linear tensor");
    }

    auto *anchorxData = dynamic_cast<const cv::ITensorDataPitchDevice *>(anchor_x.exportData());
    if (anchorxData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "anchor_x must be device-acessible, pitch-linear tensor");
    }

    auto *anchoryData = dynamic_cast<const cv::ITensorDataPitchDevice *>(anchor_y.exportData());
    if (anchoryData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "anchor_y must be device-acessible, pitch-linear tensor");
    }

    auto *erasingwData = dynamic_cast<const cv::ITensorDataPitchDevice *>(erasing_w.exportData());
    if (erasingwData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "erasing_w must be device-acessible, pitch-linear tensor");
    }

    auto *erasinghData = dynamic_cast<const cv::ITensorDataPitchDevice *>(erasing_h.exportData());
    if (erasinghData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "erasing_h must be device-acessible, pitch-linear tensor");
    }

    auto *erasingcData = dynamic_cast<const cv::ITensorDataPitchDevice *>(erasing_c.exportData());
    if (erasingcData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "erasing_c must be device-acessible, pitch-linear tensor");
    }

    auto *valuesData = dynamic_cast<const cv::ITensorDataPitchDevice *>(values.exportData());
    if (valuesData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "values must be device-acessible, pitch-linear tensor");
    }

    auto *imgIdxData = dynamic_cast<const cv::ITensorDataPitchDevice *>(imgIdx.exportData());
    if (imgIdxData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "imgIdx must be device-acessible, pitch-linear tensor");
    }

    leg::helpers::CheckOpErrThrow(m_legacyOp->infer(*inData, *outData, *anchorxData, *anchoryData, *erasingwData, *erasinghData, *erasingcData, *valuesData, *imgIdxData, max_eh, max_ew, random, seed, inplace, stream));
}


nv::cv::priv::Version Erase::doGetVersion() const
{
    // TODO: How to decouple NVCV version from legacy version?
    return nv::cv::priv::CURRENT_VERSION;
}

} // namespace nv::cvop::priv