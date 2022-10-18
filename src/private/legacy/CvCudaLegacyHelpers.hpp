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
 * @file CvCuda:egacyHelpers.hpp
 *
 * @brief Defines util functions for conversion between nvcv and legacy cv cuda
 */

#ifndef CV_CUDA_LEGACY_HELPERS_HPP
#define CV_CUDA_LEGACY_HELPERS_HPP

#include "CvCudaLegacy.h"

#include "CvCudaUtils.cuh"

#include <nvcv/IImage.hpp>
#include <private/core/Exception.hpp>
#include <private/fmt/ImageFormat.hpp>

namespace nv::cv::legacy::helpers {

cuda_op::DataFormat GetLegacyDataFormat(int32_t numberChannels, int32_t numberPlanes, int32_t numberInBatch);

cuda_op::DataType GetLegacyDataType(int32_t bpc, cv::DataType type);

cuda_op::DataType   GetLegacyDataType(priv::ImageFormat fmt);
cuda_op::DataFormat GetLegacyDataFormat(priv::ImageFormat fmt, int32_t numberInBatch);

cuda_op::DataType   GetLegacyDataType(ImageFormat fmt);
cuda_op::DataFormat GetLegacyDataFormat(ImageFormat fmt, int32_t numberInBatch);
cuda_op::DataFormat GetLegacyDataFormat(TensorLayout layout);

template<typename data_type>
inline void PopulatePlaneArrayFromPitch(legacy::cuda_op::PlaneArray<data_type> &planes, const NVCVImageData &imageData)
{
    for (uint32_t i = 0; i < imageData.buffer.pitch.numPlanes; ++i)
    {
        planes.planes[i] = (data_type *)imageData.buffer.pitch.planes[i].buffer;
    }
};

inline void CheckOpErrThrow(cuda_op::ErrorCode status)
{
    // This check gets inlined easier, and it's normal code path.
    if (status != cuda_op::ErrorCode::SUCCESS)
    {
        throw util::Exception(NVCV_ERROR_INTERNAL, "Internal Error from operator =%d", status);
    }
}

} // namespace nv::cv::legacy::helpers

#endif // CV_CUDA_LEGACY_HELPERS_HPP
