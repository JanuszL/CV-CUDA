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

/**
 * @file CvCudaLegacyHelpers.hpp
 *
 * @brief Defines util functions for conversion between nvcv and legacy cv cuda
 */

#ifndef CV_CUDA_LEGACY_HELPERS_HPP
#define CV_CUDA_LEGACY_HELPERS_HPP

#include "CvCudaLegacy.h"

#include <nvcv/Exception.hpp>
#include <nvcv/IImage.hpp>
#include <nvcv/TensorShapeInfo.hpp>

namespace nv::cv::legacy::helpers {

cuda_op::DataFormat GetLegacyDataFormat(int32_t numberChannels, int32_t numberPlanes, int32_t numberInBatch);

cuda_op::DataType GetLegacyDataType(int32_t bpc, cv::DataType type);
cuda_op::DataType GetLegacyDataType(PixelType dtype);
cuda_op::DataType GetLegacyDataType(ImageFormat fmt);

cuda_op::DataFormat GetLegacyDataFormat(const TensorLayout &layout);
cuda_op::DataFormat GetLegacyDataFormat(const IImageBatchVarShapeDataPitchDevice &imgBatch);
cuda_op::DataFormat GetLegacyDataFormat(const ITensorDataPitchDevice &tensor);

cuda_op::DataShape GetLegacyDataShape(const TensorShapeInfoImage &shapeInfo);

Size2D GetMaxImageSize(const ITensorDataPitchDevice &tensor);
Size2D GetMaxImageSize(const IImageBatchVarShapeDataPitchDevice &imageBatch);

inline void CheckOpErrThrow(cuda_op::ErrorCode status)
{
    // This check gets inlined easier, and it's normal code path.
    if (status != cuda_op::ErrorCode::SUCCESS)
    {
        throw Exception(Status::ERROR_INTERNAL, "Internal Error from operator =%d", status);
    }
}

} // namespace nv::cv::legacy::helpers

#endif // CV_CUDA_LEGACY_HELPERS_HPP
