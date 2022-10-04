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

#ifndef NVCV_IIMAGE_HPP
#define NVCV_IIMAGE_HPP

#include "IImageData.hpp"
#include "Image.h"
#include "ImageFormat.hpp"
#include "Size.hpp"
#include "alloc/IAllocator.hpp"

namespace nv { namespace cv {

class IImage
{
public:
    virtual ~IImage() = default;

    NVCVImage handle() const;

    Size2D      size() const;
    ImageFormat format() const;

    IAllocator &alloc() const;

    const IImageData *exportData() const;

private:
    // NVI idiom
    virtual NVCVImage         doGetHandle() const  = 0;
    virtual Size2D            doGetSize() const    = 0;
    virtual ImageFormat       doGetFormat() const  = 0;
    virtual IAllocator       &doGetAlloc() const   = 0;
    virtual const IImageData *doExportData() const = 0;
};

class IImageWrapData : public virtual IImage
{
public:
    void setData(const IImageData &data);
    void resetData();

private:
    virtual void doSetData(const IImageData &data) = 0;
    virtual void doResetData()                     = 0;
};

// Implementation ------------------------------------

inline NVCVImage IImage::handle() const
{
    NVCVImage h = doGetHandle();
    assert(h != nullptr && "Post-condition failed");
    return h;
}

inline Size2D IImage::size() const
{
    Size2D size = doGetSize();

    assert(size.w >= 0 && "Post-condition failed");
    assert(size.h >= 0 && "Post-condition failed");

    return size;
}

inline ImageFormat IImage::format() const
{
    return doGetFormat();
}

inline IAllocator &IImage::alloc() const
{
    return doGetAlloc();
}

inline const IImageData *IImage::exportData() const
{
    return doExportData();
}

void IImageWrapData::setData(const IImageData &data)
{
    doSetData(data);
}

void IImageWrapData::resetData()
{
    doResetData();
}

}} // namespace nv::cv

#endif // NVCV_IIMAGE_HPP
