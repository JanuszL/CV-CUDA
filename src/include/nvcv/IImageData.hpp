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
#include "detail/CudaFwd.h"

namespace nv { namespace cv {

// Interface hierarchy of image contents
class IImageData
{
public:
    virtual ~IImageData() = default;

    ImageFormat format() const;

    const NVCVImageData &cdata() const;

private:
    // NVI idiom
    virtual ImageFormat          doGetFormat() const = 0;
    virtual const NVCVImageData &doGetCData() const  = 0;
};

class IImageDataCudaArray : public IImageData
{
public:
    int         numPlanes() const;
    cudaArray_t plane(int p) const;

private:
    virtual int         doGetNumPlanes() const  = 0;
    virtual cudaArray_t doGetPlane(int p) const = 0;
};

using ImagePlanePitch = NVCVImagePlanePitch;

class IImageDataPitchDevice : public IImageData
{
public:
    Size2D size() const;

    int                    numPlanes() const;
    const ImagePlanePitch &plane(int p) const;

private:
    virtual Size2D doGetSize() const = 0;

    virtual int                    doGetNumPlanes() const  = 0;
    virtual const ImagePlanePitch &doGetPlane(int p) const = 0;
};

// Implementation - IImageData
inline ImageFormat IImageData::format() const
{
    return doGetFormat();
}

inline const NVCVImageData &IImageData::cdata() const
{
    return doGetCData();
}

// Implementation - IImageDataCudaArray
inline int32_t IImageDataCudaArray::numPlanes() const
{
    int32_t planes = doGetNumPlanes();
    assert(planes == this->format().numPlanes() && "Post-condition failed");
    return planes;
}

inline cudaArray_t IImageDataCudaArray::plane(int p) const
{
    assert(0 <= p && p < this->numPlanes() && "Pre-condition failed");
    return doGetPlane(p);
}

// Implementation - IImageDataPitchDevice
inline Size2D IImageDataPitchDevice::size() const
{
    Size2D size = doGetSize();
    assert(size.w >= 0 && "Post-condition failed");
    assert(size.h >= 0 && "Post-condition failed");
    return size;
}

inline int32_t IImageDataPitchDevice::numPlanes() const
{
    int32_t np = doGetNumPlanes();
    assert(np >= 0 && "Post-condition failed");
    return np;
}

inline const ImagePlanePitch &IImageDataPitchDevice::plane(int p) const
{
    assert(0 <= p && p < this->numPlanes() && "Pre-condition failed");
    return doGetPlane(p);
}

}} // namespace nv::cv

#endif // NVCV_IIMAGEDATA_HPP
