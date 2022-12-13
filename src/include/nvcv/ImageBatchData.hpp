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

#ifndef NVCV_IMAGEBATCHDATA_HPP
#define NVCV_IMAGEBATCHDATA_HPP

#include "IImageBatchData.hpp"

namespace nv { namespace cv {

// ImageBatchVarShapeDataPitchDevice definition -----------------------

class ImageBatchVarShapeDataPitchDevice : public IImageBatchVarShapeDataPitchDevice
{
public:
    using Buffer = NVCVImageBatchVarShapeBufferPitch;

    explicit ImageBatchVarShapeDataPitchDevice(int32_t numImages, const Buffer &buffer);
    explicit ImageBatchVarShapeDataPitchDevice(const NVCVImageBatchData &data);
};

}} // namespace nv::cv

#include "detail/ImageBatchDataImpl.hpp"

#endif // NVCV_IMAGEBATCHDATA_HPP
