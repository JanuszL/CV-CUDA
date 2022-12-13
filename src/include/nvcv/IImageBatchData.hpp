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
    virtual ~IImageBatchData() = 0;

    int32_t numImages() const;

    const NVCVImageBatchData &cdata() const;

protected:
    IImageBatchData() = default;
    IImageBatchData(const NVCVImageBatchData &data);

    NVCVImageBatchData &cdata();

private:
    NVCVImageBatchData m_data;
};

class IImageBatchVarShapeData : public IImageBatchData
{
public:
    virtual ~IImageBatchVarShapeData() = 0;

    const NVCVImageFormat *formatList() const;
    const NVCVImageFormat *hostFormatList() const;
    Size2D                 maxSize() const;
    ImageFormat            uniqueFormat() const;

protected:
    using IImageBatchData::IImageBatchData;
};

class IImageBatchVarShapeDataPitch : public IImageBatchVarShapeData
{
public:
    virtual ~IImageBatchVarShapeDataPitch() = 0;

    const NVCVImageBufferPitch *imageList() const;

protected:
    using IImageBatchVarShapeData::IImageBatchVarShapeData;
};

class IImageBatchVarShapeDataPitchDevice : public IImageBatchVarShapeDataPitch
{
public:
    virtual ~IImageBatchVarShapeDataPitchDevice() = 0;

protected:
    using IImageBatchVarShapeDataPitch::IImageBatchVarShapeDataPitch;
};

}} // namespace nv::cv

#include "detail/IImageBatchDataImpl.hpp"

#endif // NVCV_IIMAGEBATCHDATA_HPP
