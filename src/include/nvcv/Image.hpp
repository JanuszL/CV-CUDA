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

#ifndef NVCV_IMAGE_HPP
#define NVCV_IMAGE_HPP

#include "IImage.hpp"
#include "ImageData.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"
#include "alloc/IAllocator.hpp"

#include <functional>
#include <type_traits>

namespace nv { namespace cv {

// Image definition -------------------------------------
// Image allocated by cv-cuda
class Image : public IImage
{
public:
    using Requirements = NVCVImageRequirements;
    static Requirements CalcRequirements(const Size2D &size, ImageFormat fmt, const MemAlignment &bufAlign = {});

    explicit Image(const Requirements &reqs, IAllocator *alloc = nullptr);
    explicit Image(const Size2D &size, ImageFormat fmt, IAllocator *alloc = nullptr, const MemAlignment &bufAlign = {});
    ~Image();

    Image(const Image &) = delete;

private:
    NVCVImageHandle doGetHandle() const final;

    NVCVImageHandle m_handle;
};

// ImageWrapData definition -------------------------------------
// Image that wraps an image data allocated outside cv-cuda

using ImageDataCleanupFunc = void(const IImageData &);

class ImageWrapData : public IImage
{
public:
    explicit ImageWrapData(const IImageData &data, std::function<ImageDataCleanupFunc> cleanup = nullptr);
    ~ImageWrapData();

    ImageWrapData(const Image &) = delete;

private:
    NVCVImageHandle doGetHandle() const final;

    static void doCleanup(void *ctx, const NVCVImageData *data);

    NVCVImageHandle                     m_handle;
    std::function<ImageDataCleanupFunc> m_cleanup;
};

// ImageWrapHandle definition -------------------------------------
// Refers to an external NVCVImageHandle. It doesn't own it.
class ImageWrapHandle : public IImage
{
public:
    explicit ImageWrapHandle(NVCVImageHandle handle);

    ImageWrapHandle(const ImageWrapHandle &that);

private:
    NVCVImageHandle doGetHandle() const final;

    NVCVImageHandle m_handle;
};

}} // namespace nv::cv

#include "detail/ImageImpl.hpp"

#endif // NVCV_IMAGE_HPP
