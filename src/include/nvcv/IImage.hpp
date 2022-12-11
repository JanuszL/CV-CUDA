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
#include "ImageData.hpp"
#include "ImageFormat.hpp"
#include "Size.hpp"

#include <type_traits>

namespace nv { namespace cv {

class IImage
{
public:
    virtual ~IImage();

    NVCVImageHandle handle() const;

    Size2D      size() const;
    ImageFormat format() const;

    const IImageData *exportData() const;

protected:
    IImage();

private:
    virtual NVCVImageHandle doGetHandle() const = 0;

    // Where the concrete class for exported image data will be allocated
    // Should be an std::variant in C++17.
    union Arena
    {
        ImageDataCudaArray   cudaArray;
        ImageDataPitchDevice devPitch;
    };

    mutable std::aligned_storage<sizeof(Arena), alignof(Arena)>::type m_cacheDataArena;
    mutable IImageData                                               *m_cacheDataPtr;
};

}} // namespace nv::cv

#include "detail/IImageImpl.hpp"

#endif // NVCV_IIMAGE_HPP
