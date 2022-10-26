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

#ifndef NVCV_IMAGEDATA_HPP
#define NVCV_IMAGEDATA_HPP

#include "IImageData.hpp"

namespace nv { namespace cv {

// ImageDataCudaArray definition -----------------------
class ImageDataCudaArray final : public IImageDataCudaArray
{
public:
    using Buffer = NVCVImageBufferCudaArray;

    explicit ImageDataCudaArray(ImageFormat format, const Buffer &data);

private:
    NVCVImageData m_data;

    ImageFormat          doGetFormat() const override;
    int32_t              doGetNumPlanes() const override;
    cudaArray_t          doGetPlane(int p) const override;
    const NVCVImageData &doGetCData() const override;
};

// ImageDataPitchDevice definition -----------------------
class ImageDataPitchDevice final : public IImageDataPitchDevice
{
public:
    using Buffer = NVCVImageBufferPitch;

    explicit ImageDataPitchDevice(ImageFormat format, const Buffer &data);

private:
    NVCVImageData m_data;

    Size2D      doGetSize() const override;
    ImageFormat doGetFormat() const override;

    int32_t                doGetNumPlanes() const override;
    const ImagePlanePitch &doGetPlane(int p) const override;

    const NVCVImageData &doGetCData() const override;
};

// ImageDataBlock implementation -----------------------
inline ImageDataCudaArray::ImageDataCudaArray(ImageFormat format, const Buffer &data)
{
    m_data.format           = format;
    m_data.bufferType       = NVCV_IMAGE_BUFFER_CUDA_ARRAY;
    m_data.buffer.cudaarray = data;
}

inline ImageFormat ImageDataCudaArray::doGetFormat() const
{
    return ImageFormat{m_data.format};
}

inline int32_t ImageDataCudaArray::doGetNumPlanes() const
{
    return m_data.buffer.cudaarray.numPlanes;
}

inline cudaArray_t ImageDataCudaArray::doGetPlane(int p) const
{
    return m_data.buffer.cudaarray.planes[p];
}

inline const NVCVImageData &ImageDataCudaArray::doGetCData() const
{
    return m_data;
}

// ImageDataPitchDevice implementation -----------------------
inline ImageDataPitchDevice::ImageDataPitchDevice(ImageFormat format, const Buffer &data)
{
    m_data.format       = format;
    m_data.bufferType   = NVCV_IMAGE_BUFFER_PITCH_DEVICE;
    m_data.buffer.pitch = data;
}

inline Size2D ImageDataPitchDevice::doGetSize() const
{
    Size2D out;
    if (m_data.buffer.pitch.numPlanes > 0)
    {
        out.w = m_data.buffer.pitch.planes[0].width;
        out.h = m_data.buffer.pitch.planes[0].height;
    }
    else
    {
        out = {0, 0};
    }
    return out;
}

inline ImageFormat ImageDataPitchDevice::doGetFormat() const
{
    return ImageFormat{m_data.format};
}

inline int ImageDataPitchDevice::doGetNumPlanes() const
{
    return m_data.buffer.pitch.numPlanes;
}

inline const ImagePlanePitch &ImageDataPitchDevice::doGetPlane(int p) const
{
    return m_data.buffer.pitch.planes[p];
}

inline const NVCVImageData &ImageDataPitchDevice::doGetCData() const
{
    return m_data;
}

}} // namespace nv::cv

#endif // NVCV_DETAIL_IMAGEDATA_HPP
