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

Erase::Erase(int num_erasing_area)
{
    leg::cuda_op::DataShape maxIn, maxOut;
    // maxIn/maxOut not used by op.
    m_legacyOp         = std::make_unique<leg::cuda_op::Erase>(maxIn, maxOut, num_erasing_area);
    m_legacyOpVarShape = std::make_unique<leg::cuda_op::EraseVarShape>(maxIn, maxOut, num_erasing_area);
}

void Erase::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::ITensor &anchor,
                       cv::ITensor &erasing, cv::ITensor &values, cv::ITensor &imgIdx, bool random,
                       unsigned int seed) const
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

    auto *anchorData = dynamic_cast<const cv::ITensorDataPitchDevice *>(anchor.exportData());
    if (anchorData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "anchor must be device-acessible, pitch-linear tensor");
    }

    auto *erasingData = dynamic_cast<const cv::ITensorDataPitchDevice *>(erasing.exportData());
    if (erasingData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "erasing must be device-acessible, pitch-linear tensor");
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

    bool inplace = (in.handle() == out.handle());
    leg::helpers::CheckOpErrThrow(m_legacyOp->infer(*inData, *outData, *anchorData, *erasingData, *valuesData,
                                                    *imgIdxData, random, seed, inplace, stream));
}

void Erase::operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                       cv::ITensor &anchor, cv::ITensor &erasing, cv::ITensor &values, cv::ITensor &imgIdx, bool random,
                       unsigned int seed) const
{
    auto *anchorData = dynamic_cast<const cv::ITensorDataPitchDevice *>(anchor.exportData());
    if (anchorData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "anchor must be device-acessible, pitch-linear tensor");
    }

    auto *erasingData = dynamic_cast<const cv::ITensorDataPitchDevice *>(erasing.exportData());
    if (erasingData == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT,
                            "erasing must be device-acessible, pitch-linear tensor");
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

    bool inplace = (in.handle() == out.handle());
    leg::helpers::CheckOpErrThrow(m_legacyOpVarShape->infer(in, out, *anchorData, *erasingData, *valuesData,
                                                            *imgIdxData, random, seed, inplace, stream));
}

} // namespace nv::cvop::priv
