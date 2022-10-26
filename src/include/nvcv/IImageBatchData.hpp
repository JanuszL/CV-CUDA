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
#include "ImageFormat.hpp"
#include "detail/CudaFwd.h"

namespace nv { namespace cv {

// Interface hierarchy of image batch contents
class IImageBatchData
{
public:
    virtual ~IImageBatchData() = default;

    ImageFormat format() const;

    const NVCVImageBatchData &cdata() const;

private:
    // NVI idiom
    virtual ImageFormat               doGetFormat() const = 0;
    virtual const NVCVImageBatchData &doGetCData() const  = 0;
};

class IImageBatchVarShapeDataPitchDevice : public IImageBatchData
{
public:
    int32_t                numImages() const;
    const ImagePlanePitch *imgPlanes() const;

private:
    virtual int32_t                doGetNumImages() const   = 0;
    virtual const ImagePlanePitch *doGetImagePlanes() const = 0;
};

// Implementation - IImageBatchData
inline ImageFormat IImageBatchData::format() const
{
    return doGetFormat();
}

inline const NVCVImageBatchData &IImageBatchData::cdata() const
{
    return doGetCData();
}

// Implementation - IImageBatchVarShapeDataPitchDevice
inline int32_t IImageBatchVarShapeDataPitchDevice::numImages() const
{
    int32_t size = doGetNumImages();
    assert(size >= 0 && "Post-condition failed");
    return size;
}

inline const ImagePlanePitch *IImageBatchVarShapeDataPitchDevice::imgPlanes() const
{
    return doGetImagePlanes();
}

}} // namespace nv::cv

#endif // NVCV_IIMAGEBATCHDATA_HPP
