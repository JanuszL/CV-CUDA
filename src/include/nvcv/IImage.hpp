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

using ImageDataCleanupFunc = void(const IImageData &);

class IImageWrapData : public virtual IImage
{
public:
    // Redefines the data only.
    void resetData(const IImageData &data);
    void resetData();

    // Redefines the data and its cleanup function.
    void resetDataAndCleanup(const IImageData &data, std::function<ImageDataCleanupFunc> cleanup);
    void resetDataAndCleanup();

private:
    virtual void doResetData(const IImageData *data)                                                        = 0;
    virtual void doResetDataAndCleanup(const IImageData *data, std::function<ImageDataCleanupFunc> cleanup) = 0;
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

void IImageWrapData::resetData(const IImageData &data)
{
    doResetData(&data);
}

void IImageWrapData::resetData()
{
    doResetData(nullptr);
}

void IImageWrapData::resetDataAndCleanup(const IImageData &data, std::function<ImageDataCleanupFunc> cleanup)
{
    doResetDataAndCleanup(&data, std::move(cleanup));
}

void IImageWrapData::resetDataAndCleanup()
{
    doResetDataAndCleanup(nullptr, nullptr);
}

}} // namespace nv::cv

#endif // NVCV_IIMAGE_HPP
