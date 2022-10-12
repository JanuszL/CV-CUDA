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

#include <functional>

namespace nv { namespace cv {

class IImage
{
public:
    virtual ~IImage() = default;

    NVCVImageHandle handle() const;

    Size2D      size() const;
    ImageFormat format() const;

    IAllocator &alloc() const;

    const IImageData *exportData() const;

private:
    // NVI idiom
    virtual NVCVImageHandle   doGetHandle() const  = 0;
    virtual Size2D            doGetSize() const    = 0;
    virtual ImageFormat       doGetFormat() const  = 0;
    virtual IAllocator       &doGetAlloc() const   = 0;
    virtual const IImageData *doExportData() const = 0;
};

// Implementation ------------------------------------

inline NVCVImageHandle IImage::handle() const
{
    NVCVImageHandle h = doGetHandle();
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

}} // namespace nv::cv

#endif // NVCV_IIMAGE_HPP
