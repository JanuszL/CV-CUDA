
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

#include "OpPillowResize.hpp"

#include <nvcv/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>
#include <private/legacy/CvCudaLegacyHelpers.hpp>
#include <util/CheckError.hpp>

namespace nv::cvop::priv {

namespace leg = cv::legacy;

PillowResize::PillowResize(cv::Size2D maxSize, int maxBatchSize, NVCVImageFormat fmt)
{
    int32_t bpc[4];
    nvcvImageFormatGetBitsPerChannel(fmt, bpc);
    int32_t maxChannel = 0;
    nvcvImageFormatGetNumChannels(fmt, &maxChannel);
    NVCVDataType type;
    nvcvImageFormatGetDataType(fmt, &type);
    nv::cv::DataType        dtype     = static_cast<nv::cv::DataType>(type);
    leg::cuda_op::DataType  data_type = leg::helpers::GetLegacyDataType(bpc[0], dtype);
    leg::cuda_op::DataShape maxIn(maxBatchSize, maxChannel, maxSize.h, maxSize.w),
        maxOut(maxBatchSize, maxChannel, maxSize.h, maxSize.w);
    m_legacyOp = std::make_unique<leg::cuda_op::PillowResize>(maxIn, maxOut, data_type);
}

void PillowResize::operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out,
                              const NVCVInterpolationType interpolation) const
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

    NVCV_CHECK_THROW(m_legacyOp->infer(*inData, *outData, interpolation, stream));
}

} // namespace nv::cvop::priv
