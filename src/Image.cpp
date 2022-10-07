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
#include <nvcv/Image.hpp>
#include <private/core/Exception.hpp>
#include <private/core/IAllocator.hpp>
#include <private/core/Image.hpp>
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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageCreate,
                (const NVCVImageRequirements *reqs, NVCVAllocatorHandle halloc, NVCVImage *handle))
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

            static_assert(sizeof(NVCVImage) >= sizeof(priv::Image));
            static_assert(alignof(NVCVImage) % alignof(priv::Image) == 0);

            new (handle) priv::Image{*reqs, alloc};
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageCreateWrapData,
                (const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup, NVCVImage *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            static_assert(sizeof(NVCVImage) >= sizeof(priv::ImageWrapData));
            static_assert(alignof(NVCVImage) % alignof(priv::ImageWrapData) == 0);

            if (data)
            {
                new (handle) priv::ImageWrapData{*data, cleanup, ctxCleanup};
            }
            else
            {
                new (handle) priv::ImageWrapData{};
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageDestroy, (NVCVImage * handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (!priv::IsDestroyed(handle))
            {
                priv::ToStaticPtr<priv::IImage>(handle)->~IImage();
                memset(handle, 0, sizeof(*handle));

                NVCV_ASSERT(priv::IsDestroyed(handle));
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageGetSize, (NVCVImage * handle, int32_t *width, int32_t *height))
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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageGetFormat, (NVCVImage * handle, NVCVImageFormat *fmt))
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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageGetAllocator, (NVCVImage * handle, NVCVAllocatorHandle *halloc))
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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageGetType, (NVCVImage * handle, NVCVTypeImage *type))
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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageExportData, (NVCVImage * handle, NVCVImageData *data))
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
