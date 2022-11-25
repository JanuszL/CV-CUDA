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

#ifndef NVCV_IIMAGEBATCHDATA_HPP
#define NVCV_IIMAGEBATCHDATA_HPP

#include "ImageBatchData.h"
#include "ImageData.hpp"
#include "detail/CudaFwd.h"
#include "detail/Optional.hpp"

namespace nv { namespace cv {

// Interface hierarchy of image batch contents
class IImageBatchData
{
public:
    virtual ~IImageBatchData() = default;

    int32_t numImages() const;

    const NVCVImageBatchData &cdata() const;

private:
    // NVI idiom
    virtual int32_t                   doGetNumImages() const = 0;
    virtual const NVCVImageBatchData &doGetCData() const     = 0;
};

class IImageBatchVarShapeData : public IImageBatchData
{
public:
    const NVCVImageFormat *formatList() const;
    const NVCVImageFormat *hostFormatList() const;
    Size2D                 maxSize() const;
    ImageFormat            uniqueFormat() const;

private:
    virtual const NVCVImageFormat *doGetFormatList() const     = 0;
    virtual const NVCVImageFormat *doGetHostFormatList() const = 0;
    virtual Size2D                 doGetMaxSize() const        = 0;
    virtual ImageFormat            doGetUniqueFormat() const   = 0;
};

class IImageBatchVarShapeDataPitch : public IImageBatchVarShapeData
{
public:
    const NVCVImageBufferPitch *imageList() const;

private:
    virtual const NVCVImageBufferPitch *doGetImageList() const = 0;
};

class IImageBatchVarShapeDataPitchDevice : public IImageBatchVarShapeDataPitch
{
};

// Implementation - IImageBatchData
inline int32_t IImageBatchData::numImages() const
{
    int32_t size = doGetNumImages();
    assert(size >= 0 && "Post-condition failed");
    return size;
}

inline const NVCVImageBatchData &IImageBatchData::cdata() const
{
    return doGetCData();
}

// Implementation - IImageBatchVarShapeData
inline const NVCVImageFormat *IImageBatchVarShapeData::formatList() const
{
    return doGetFormatList();
}

inline const NVCVImageFormat *IImageBatchVarShapeData::hostFormatList() const
{
    return doGetHostFormatList();
}

inline Size2D IImageBatchVarShapeData::maxSize() const
{
    return doGetMaxSize();
}

inline ImageFormat IImageBatchVarShapeData::uniqueFormat() const
{
    return doGetUniqueFormat();
}

// Implementation - IImageBatchVarShapeDataPitch
inline const NVCVImageBufferPitch *IImageBatchVarShapeDataPitch::imageList() const
{
    return doGetImageList();
}

}} // namespace nv::cv

#endif // NVCV_IIMAGEBATCHDATA_HPP
