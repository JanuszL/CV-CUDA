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
class ImageDataCudaArray : public IImageDataCudaArray
{
public:
    using Buffer = NVCVImageBufferCudaArray;

    explicit ImageDataCudaArray(ImageFormat format, const Buffer &buffer);
    explicit ImageDataCudaArray(const NVCVImageData &data);
};

// ImageDataPitchDevice definition -----------------------
class ImageDataPitchDevice : public IImageDataPitchDevice
{
public:
    using Buffer = NVCVImageBufferPitch;

    explicit ImageDataPitchDevice(ImageFormat format, const Buffer &buffer);
    explicit ImageDataPitchDevice(const NVCVImageData &data);
};

// ImageDataPitchHost definition -----------------------
class ImageDataPitchHost : public IImageDataPitchHost
{
public:
    using Buffer = NVCVImageBufferPitch;

    explicit ImageDataPitchHost(ImageFormat format, const Buffer &buffer);
    explicit ImageDataPitchHost(const NVCVImageData &data);
};

}} // namespace nv::cv

#include "detail/ImageDataImpl.hpp"

#endif // NVCV_DETAIL_IMAGEDATA_HPP
