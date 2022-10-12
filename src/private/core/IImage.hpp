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

#ifndef NVCV_PRIV_IIMAGE_HPP
#define NVCV_PRIV_IIMAGE_HPP

#include "ICoreObject.hpp"

#include <fmt/ImageFormat.hpp>
#include <nvcv/Image.h>

namespace nv::cv::priv {

class IAllocator;

class IImage : public ICoreObjectHandle<IImage, NVCVImageHandle>
{
public:
    virtual Size2D      size() const   = 0;
    virtual ImageFormat format() const = 0;

    virtual NVCVTypeImage type() const = 0;

    virtual IAllocator &alloc() const = 0;

    virtual void exportData(NVCVImageData &data) const = 0;
};

class IImageWrapData : public IImage
{
public:
    virtual void setData(const NVCVImageData *data)                                                               = 0;
    virtual void setDataAndCleanup(const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup) = 0;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_IIMAGE_HPP
