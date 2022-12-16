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
