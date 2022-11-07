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

#include <nvcv/PixelType.hpp>
#include <private/core/Exception.hpp>
#include <private/fmt/PixelType.hpp>
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

static cuda_op::DataType GetLegacyCvFloatType(int32_t bpc)
{
    if (bpc == 64)
        return cuda_op::DataType::kCV_64F;
    if (bpc == 32)
        return cuda_op::DataType::kCV_32F;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16F;

    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for float cuda op type ", bpc);
}

static cuda_op::DataType GetLegacyCvSignedType(int32_t bpc)
{
    if (bpc == 8)
        return cuda_op::DataType::kCV_8S;
    if (bpc == 16)
        return cuda_op::DataType::kCV_16S;
    if (bpc == 32)
        return cuda_op::DataType::kCV_32S;

    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Invalid bpc(%d) for signed cuda op type ", bpc);
}

static cuda_op::DataType GetLegacyCvUnsignedType(int32_t bpc)
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
        return GetLegacyCvFloatType(bpc);
    }

    case cv::DataType::SIGNED:
    {
        return GetLegacyCvSignedType(bpc);
    }

    case cv::DataType::UNSIGNED:
    {
        return GetLegacyCvUnsignedType(bpc);
    }
    }
    throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Only planar formats supported ");
}

cuda_op::DataType GetLegacyDataType(PixelType dtype_)
{
    priv::PixelType dtype{dtype_}; // to avoid using public API

    auto bpc = dtype.bpc();

    for (int i = 1; i < dtype.numChannels(); ++i)
    {
        if (bpc[i] != bpc[0])
        {
            throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "All channels must have same bit-depth");
        }
    }

    return GetLegacyDataType(dtype.bpc()[0], (cv::DataType)dtype.dataType());
}

cuda_op::DataType GetLegacyDataType(ImageFormat fmt_)
{
    priv::ImageFormat fmt{fmt_}; // to avoid using public API

    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planePixelType(i) != fmt.planePixelType(0))
        {
            throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "All planes must have the same pixel type");
        }
    }

    return GetLegacyDataType(PixelType{fmt.planePixelType(0).value()});
}

cuda_op::DataShape GetLegacyDataShape(const TensorShapeInfoImage &shapeInfo)
{
    return cuda_op::DataShape(shapeInfo.numSamples(), shapeInfo.numChannels(), shapeInfo.numRows(),
                              shapeInfo.numCols());
}

cuda_op::DataFormat GetLegacyDataFormat(const IImageBatchVarShapeDataPitchDevice &imgBatch)
{
    priv::ImageFormat fmt{imgBatch.format()}; // to avoid using public API

    for (int i = 1; i < fmt.numPlanes(); ++i)
    {
        if (fmt.planePixelType(i) != fmt.planePixelType(0))
        {
            throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "All planes must have the same pixel type");
        }
    }

    if (fmt.numPlanes() >= 2)
    {
        if (fmt.numPlanes() != fmt.numChannels())
        {
            throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Planar images must have one channel per plane");
        }

        if (imgBatch.numImages() >= 2)
        {
            return legacy::cuda_op::DataFormat::kNCHW;
        }
        else
        {
            return legacy::cuda_op::DataFormat::kCHW;
        }
    }
    else
    {
        if (imgBatch.numImages() >= 2)
        {
            return legacy::cuda_op::DataFormat::kNHWC;
        }
        else
        {
            return legacy::cuda_op::DataFormat::kHWC;
        }
    }
}

cuda_op::DataFormat GetLegacyDataFormat(const TensorLayout &layout)
{
    if (layout == TensorLayout::NCHW)
    {
        return legacy::cuda_op::DataFormat::kNCHW;
    }
    else if (layout == TensorLayout::CHW)
    {
        return legacy::cuda_op::DataFormat::kCHW;
    }
    else if (layout == TensorLayout::NHWC)
    {
        return legacy::cuda_op::DataFormat::kNHWC;
    }
    else if (layout == TensorLayout::HWC)
    {
        return legacy::cuda_op::DataFormat::kHWC;
    }
    else
    {
        throw util::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Tensor layout not supported");
    }
}

} // namespace nv::cv::legacy::helpers
