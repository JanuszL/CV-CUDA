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

#ifndef NVCV_IIMAGEDATA_HPP
#define NVCV_IIMAGEDATA_HPP

#include "ImageData.h"
#include "ImageFormat.hpp"
#include "Size.hpp"

namespace nv { namespace cv {

// Interface hierarchy of image contents
class IImageData
{
public:
    virtual ~IImageData() = 0;

    ImageFormat format() const;

    const NVCVImageData &cdata() const;

protected:
    IImageData() = default;
    IImageData(const NVCVImageData &data);

    NVCVImageData &cdata();

private:
    NVCVImageData m_data;
};

class IImageDataCudaArray : public IImageData
{
public:
    virtual ~IImageDataCudaArray() = 0;

    int         numPlanes() const;
    cudaArray_t plane(int p) const;

protected:
    using IImageData::IImageData;
};

using ImagePlanePitch = NVCVImagePlanePitch;

class IImageDataPitch : public IImageData
{
public:
    virtual ~IImageDataPitch() = 0;

    Size2D size() const;

    int                    numPlanes() const;
    const ImagePlanePitch &plane(int p) const;

protected:
    using IImageData::IImageData;
};

class IImageDataPitchDevice : public IImageDataPitch
{
public:
    virtual ~IImageDataPitchDevice() = 0;

protected:
    using IImageDataPitch::IImageDataPitch;
};

class IImageDataPitchHost : public IImageDataPitch
{
public:
    virtual ~IImageDataPitchHost() = 0;

protected:
    using IImageDataPitch::IImageDataPitch;
};

}} // namespace nv::cv

#include "detail/IImageDataImpl.hpp"

#endif // NVCV_IIMAGEDATA_HPP
