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

#include "CvCudaLegacyHelpers.hpp"

#include <private/core/Exception.hpp>
#include <private/legacy/CvCudaLegacy.h>

#include <iostream>
using namespace std;

namespace nv::cv::legacy::helpers {

cuda_op::DataFormat GetLegacyDataFormat(int32_t numberChannels, int32_t numberPlanes, int32_t numberInBatch)
{
    if (numberPlanes == 1) // test for packed
    {
        return ((numberInBatch > 1) ? legacy::cuda_op::DataFormat::kNHWC : legacy::cuda_op::DataFormat::kHWC);
    }
    if (numberChannels == numberPlanes) //test for planar
    {
        return ((numberInBatch > 1) ? legacy::cuda_op::DataFormat::kNCHW : legacy::cuda_op::DataFormat::kCHW);
    }

    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                          "Only planar or packed formats supported CH = %d, planes = %d, batch = %d", numberChannels,
                          numberPlanes, numberInBatch);
}

static cuda_op::DataType getLegacyCvFloatType(int32_t bpc)
{
    if (bpc == 64)
        return cuda_op::DataType::kCV_64F;
    if (bpc == 32)
        return cuda_op::DataType::kCV_32F;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16F;

    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);
}

static cuda_op::DataType getLegacyCvSignedType(int32_t bpc)
{
    if (bpc == 8)
        return cuda_op::DataType::kCV_8S;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16S;
    if (bpc == 32)
        return cuda_op::DataType::kCV_32S;

    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);
}

static cuda_op::DataType getLegacyCvUnsignedType(int32_t bpc)
{
    if (bpc == 8)
        return cuda_op::DataType::kCV_8U;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16U;

    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for unsigned cuda op type ", bpc);
}

cuda_op::DataType GetLegacyDataType(int32_t bpc, cv::DataType type)
{
    switch (type)
    {
    case cv::DataType::FLOAT:
    {
        return getLegacyCvFloatType(bpc);
    }

    case cv::DataType::SIGNED:
    {
        return getLegacyCvSignedType(bpc);
    }

    case cv::DataType::UNSIGNED:
    {
        return getLegacyCvUnsignedType(bpc);
    }
    }
    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Only planar formats supported ");
}

cuda_op::DataType GetLegacyDataType(priv::ImageFormat fmt)
{
    std::array<int, 4> bpc = fmt.bpc();
    for (int i = 1; i < fmt.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                  "Channels in image format must have the same number of bits");
        }
    }

    return GetLegacyDataType(bpc[0], (cv::DataType)fmt.dataType());
}

cuda_op::DataFormat GetLegacyDataFormat(priv::ImageFormat fmt, int32_t numberInBatch)
{
    return GetLegacyDataFormat(fmt.numChannels(), fmt.numPlanes(), numberInBatch);
}

cuda_op::DataType GetLegacyDataType(ImageFormat fmt)
{
    return GetLegacyDataType(priv::ImageFormat{fmt.cvalue()});
}

cuda_op::DataFormat GetLegacyDataFormat(ImageFormat fmt, int32_t numberInBatch)
{
    return GetLegacyDataFormat(priv::ImageFormat(fmt.cvalue()), numberInBatch);
}

cuda_op::DataFormat GetLegacyDataFormat(TensorLayout layout)
{
    switch (layout)
    {
    case TensorLayout::NCHW:
        return legacy::cuda_op::DataFormat::kNCHW;

    case TensorLayout::NHWC:
        return legacy::cuda_op::DataFormat::kNHWC;

    default:
        throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
    }
}

} // namespace nv::cv::legacy::helpers
