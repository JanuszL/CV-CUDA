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

#include <nvcv/Image.h>
#include <private/core/AllocatorManager.hpp>
#include <private/core/Exception.hpp>
#include <private/core/IAllocator.hpp>
#include <private/core/Image.hpp>
#include <private/core/ImageManager.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageCalcRequirements,
                (int32_t width, int32_t height, NVCVImageFormat format, NVCVImageRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            *reqs = priv::Image::CalcRequirements({width, height}, priv::ImageFormat{format});
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageConstruct,
                (const NVCVImageRequirements *reqs, NVCVAllocatorHandle halloc, NVCVImageHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to image requirements must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            *handle = priv::CreateCoreObject<priv::Image>(*reqs, alloc);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageWrapDataConstruct,
                (const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup,
                 NVCVImageHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Image data must not be NULL");
            }

            *handle = priv::CreateCoreObject<priv::ImageWrapData>(*data, cleanup, ctxCleanup);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageDestroy, (NVCVImageHandle handle))
{
    return priv::ProtectCall([&] { priv::DestroyCoreObject(handle); });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetSize, (NVCVImageHandle handle, int32_t *width, int32_t *height))
{
    return priv::ProtectCall(
        [&]
        {
            if (width == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output width cannot be NULL");
            }
            if (height == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output height cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            priv::Size2D size = img.size();

            *width  = size.w;
            *height = size.h;
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetFormat, (NVCVImageHandle handle, NVCVImageFormat *fmt))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            *fmt = img.format().value();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetAllocator, (NVCVImageHandle handle, NVCVAllocatorHandle *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output allocator handle cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            *halloc = img.alloc().handle();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageGetType, (NVCVImageHandle handle, NVCVTypeImage *type))
{
    return priv::ProtectCall(
        [&]
        {
            if (type == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image type cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);

            *type = img.type();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvImageExportData, (NVCVImageHandle handle, NVCVImageData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image data cannot be NULL");
            }

            auto &img = priv::ToStaticRef<const priv::IImage>(handle);
            img.exportData(*data);
        });
}
