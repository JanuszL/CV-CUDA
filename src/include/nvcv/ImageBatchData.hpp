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

// ImageBatchVarShapeDataDevicePitch definition -----------------------
class ImageBatchVarShapeDataDevicePitch final : public IImageBatchVarShapeDataDevicePitch
{
public:
    using Buffer = NVCVImageBatchVarShapeBufferPitch;

    explicit ImageBatchVarShapeDataDevicePitch(ImageFormat format, const Buffer &data);

private:
    NVCVImageBatchData m_data;

    int32_t     doGetNumImages() const override;
    ImageFormat doGetFormat() const override;

    const ImagePlanePitch *doGetImagePlanes() const override;

    const NVCVImageBatchData &doGetCData() const override;
};

// ImageBatchVarShapeDataDevicePitch implementation -----------------------
inline ImageBatchVarShapeDataDevicePitch::ImageBatchVarShapeDataDevicePitch(ImageFormat format, const Buffer &data)
{
    m_data.format               = format;
    m_data.bufferType           = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_DEVICE_PITCH;
    m_data.buffer.varShapePitch = data;
}

inline int32_t ImageBatchVarShapeDataDevicePitch::doGetNumImages() const
{
    return m_data.buffer.varShapePitch.numImages;
}

inline ImageFormat ImageBatchVarShapeDataDevicePitch::doGetFormat() const
{
    return ImageFormat{m_data.format};
}

inline const ImagePlanePitch *ImageBatchVarShapeDataDevicePitch::doGetImagePlanes() const
{
    return m_data.buffer.varShapePitch.imgPlanes;
}

inline const NVCVImageBatchData &ImageBatchVarShapeDataDevicePitch::doGetCData() const
{
    return m_data;
}

}} // namespace nv::cv

#endif // NVCV_IMAGEBATCHDATA_HPP
