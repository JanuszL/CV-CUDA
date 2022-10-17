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

#include <nvcv/ImageBatch.h>
#include <nvcv/ImageBatch.hpp>
#include <private/core/Exception.hpp>
#include <private/core/IAllocator.hpp>
#include <private/core/ImageBatchVarShape.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/fmt/ImageFormat.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapeCalcRequirements,
                (int32_t capacity, NVCVImageFormat format, NVCVImageBatchVarShapeRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            *reqs = priv::ImageBatchVarShape::CalcRequirements(capacity, priv::ImageFormat{format});
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapeConstruct,
                (const NVCVImageBatchVarShapeRequirements *reqs, NVCVAllocatorHandle halloc,
                 NVCVImageBatchStorage *storage, NVCVImageBatchHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to varshape image batch requirements must not be NULL");
            }

            if (storage == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to image batch storage must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            static_assert(sizeof(NVCVImageBatchStorage) >= sizeof(priv::ImageBatchVarShape));
            static_assert(alignof(NVCVImageBatchStorage) % alignof(priv::ImageBatchVarShape) == 0);

            *handle = reinterpret_cast<NVCVImageBatchHandle>(new (storage) priv::ImageBatchVarShape{*reqs, alloc});
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchDestroy, (NVCVImageBatchHandle handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (!priv::IsDestroyed(handle))
            {
                priv::ToStaticPtr<priv::IImageBatch>(handle)->~IImageBatch();
                memset(handle, 0, sizeof(NVCVImageBatchStorage));

                NVCV_ASSERT(priv::IsDestroyed(handle));
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchGetSize, (NVCVImageBatchHandle handle, int32_t *size))
{
    return priv::ProtectCall(
        [&]
        {
            if (size == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output width cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *size = batch.size();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchGetCapacity, (NVCVImageBatchHandle handle, int32_t *capacity))
{
    return priv::ProtectCall(
        [&]
        {
            if (capacity == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output width cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *capacity = batch.capacity();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchGetFormat, (NVCVImageBatchHandle handle, NVCVImageFormat *fmt))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *fmt = batch.format().value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchGetAllocator,
                (NVCVImageBatchHandle handle, NVCVAllocatorHandle *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output allocator handle cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *halloc = batch.alloc().handle();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchGetType, (NVCVImageBatchHandle handle, NVCVTypeImageBatch *type))
{
    return priv::ProtectCall(
        [&]
        {
            if (type == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image batch type cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);

            *type = batch.type();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchExportData,
                (NVCVImageBatchHandle handle, CUstream stream, NVCVImageBatchData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image batch data cannot be NULL");
            }

            auto &batch = priv::ToStaticRef<const priv::IImageBatch>(handle);
            batch.exportData(stream, *data);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapePushImages,
                (NVCVImageBatchHandle handle, const NVCVImageHandle *images, int32_t numImages))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.pushImages(images, numImages);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapePushImagesCallback,
                (NVCVImageBatchHandle handle, NVCVPushImageFunc cbPushImage, void *ctxCallback))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.pushImages(cbPushImage, ctxCallback);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapePopImages, (NVCVImageBatchHandle handle, int32_t numImages))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.popImages(numImages);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapeClear, (NVCVImageBatchHandle handle))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<priv::IImageBatchVarShape>(handle);

            batch.clear();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvImageBatchVarShapeGetImages,
                (NVCVImageBatchHandle handle, int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages))
{
    return priv::ProtectCall(
        [&]
        {
            auto &batch = priv::ToDynamicRef<const priv::IImageBatchVarShape>(handle);

            batch.getImages(begIndex, outImages, numImages);
        });
}
