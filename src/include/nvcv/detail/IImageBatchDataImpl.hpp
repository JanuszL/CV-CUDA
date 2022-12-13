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

#ifndef NVCV_IIMAGEBATCHDATA_IMPL_HPP
#define NVCV_IIMAGEBATCHDATA_IMPL_HPP

#ifndef NVCV_IIMAGEBATCHDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// Implementation - IImageBatchData

inline IImageBatchData::IImageBatchData(const NVCVImageBatchData &data)
    : m_data(data)
{
}

inline IImageBatchData::~IImageBatchData()
{
    // required dtor implementation
}

inline int32_t IImageBatchData::numImages() const
{
    return this->cdata().numImages;
}

inline const NVCVImageBatchData &IImageBatchData::cdata() const
{
    return m_data;
}

inline NVCVImageBatchData &IImageBatchData::cdata()
{
    return m_data;
}

// Implementation - IImageBatchVarShapeData

inline IImageBatchVarShapeData::~IImageBatchVarShapeData()
{
    // required dtor implementation
}

inline const NVCVImageFormat *IImageBatchVarShapeData::formatList() const
{
    return this->cdata().buffer.varShapePitch.formatList;
}

inline const NVCVImageFormat *IImageBatchVarShapeData::hostFormatList() const
{
    return this->cdata().buffer.varShapePitch.hostFormatList;
}

inline Size2D IImageBatchVarShapeData::maxSize() const
{
    const NVCVImageBatchVarShapeBufferPitch &buffer = this->cdata().buffer.varShapePitch;

    return {buffer.maxWidth, buffer.maxHeight};
}

inline ImageFormat IImageBatchVarShapeData::uniqueFormat() const
{
    return ImageFormat{this->cdata().buffer.varShapePitch.uniqueFormat};
}

// Implementation - IImageBatchVarShapeDataPitch

inline IImageBatchVarShapeDataPitch::~IImageBatchVarShapeDataPitch()
{
    // required dtor implementation
}

inline const NVCVImageBufferPitch *IImageBatchVarShapeDataPitch::imageList() const
{
    return this->cdata().buffer.varShapePitch.imageList;
}

// Implementation - IImageBatchVarShapeDataPitchDevice

inline IImageBatchVarShapeDataPitchDevice::~IImageBatchVarShapeDataPitchDevice()
{
    // required dtor implementation
}

}} // namespace nv::cv

#endif // NVCV_IIMAGEBATCHDATA_IMPL_HPP
