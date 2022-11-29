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
class ImageBatchVarShapeDataPitchDevice final : public IImageBatchVarShapeDataPitchDevice
{
public:
    using Buffer = NVCVImageBatchVarShapeBufferPitch;

    explicit ImageBatchVarShapeDataPitchDevice(int32_t numImages, const Buffer &data);

private:
    NVCVImageBatchData m_data;

    int32_t     doGetNumImages() const override;
    ImageFormat doGetUniqueFormat() const override;
    Size2D      doGetMaxSize() const override;

    const NVCVImageBufferPitch *doGetImageList() const override;
    const NVCVImageFormat      *doGetFormatList() const override;
    const NVCVImageFormat      *doGetHostFormatList() const override;

    const NVCVImageBatchData &doGetCData() const override;
};

// ImageBatchVarShapeDataPitchDevice implementation -----------------------
inline ImageBatchVarShapeDataPitchDevice::ImageBatchVarShapeDataPitchDevice(int32_t numImages, const Buffer &data)
{
    m_data.numImages            = numImages;
    m_data.bufferType           = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_PITCH_DEVICE;
    m_data.buffer.varShapePitch = data;
}

inline int32_t ImageBatchVarShapeDataPitchDevice::doGetNumImages() const
{
    return m_data.numImages;
}

inline Size2D ImageBatchVarShapeDataPitchDevice::doGetMaxSize() const
{
    return {m_data.buffer.varShapePitch.maxWidth, m_data.buffer.varShapePitch.maxHeight};
}

inline ImageFormat ImageBatchVarShapeDataPitchDevice::doGetUniqueFormat() const
{
    return ImageFormat{m_data.buffer.varShapePitch.uniqueFormat};
}

inline const NVCVImageFormat *ImageBatchVarShapeDataPitchDevice::doGetFormatList() const
{
    return m_data.buffer.varShapePitch.formatList;
}

inline const NVCVImageFormat *ImageBatchVarShapeDataPitchDevice::doGetHostFormatList() const
{
    return m_data.buffer.varShapePitch.hostFormatList;
}

inline const NVCVImageBufferPitch *ImageBatchVarShapeDataPitchDevice::doGetImageList() const
{
    return m_data.buffer.varShapePitch.imageList;
}

inline const NVCVImageBatchData &ImageBatchVarShapeDataPitchDevice::doGetCData() const
{
    return m_data;
}

}} // namespace nv::cv

#endif // NVCV_IMAGEBATCHDATA_HPP
