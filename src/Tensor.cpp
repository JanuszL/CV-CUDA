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

#include <nvcv/Tensor.h>
#include <nvcv/Tensor.hpp>
#include <private/core/Exception.hpp>
#include <private/core/IAllocator.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/Tensor.hpp>
#include <private/core/TensorLayout.hpp>
#include <private/core/TensorWrapData.hpp>
#include <private/fmt/ImageFormat.hpp>

#include <algorithm>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorCalcRequirementsForImages,
                (int32_t batch, int32_t width, int32_t height, NVCVImageFormat format, NVCVTensorRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            priv::ImageFormat fmt{format};

            *reqs = priv::Tensor::CalcRequirements(batch, {width, height}, fmt);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorCalcRequirementsNCHW,
                (int32_t nbatch, int32_t channels, int32_t height, int32_t width, NVCVImageFormat format,
                 NVCVTensorRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            priv::ImageFormat fmt{format};

            *reqs = priv::Tensor::CalcRequirements(priv::DimsNCHW{nbatch, channels, height, width}, fmt);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorConstruct,
                (const NVCVTensorRequirements *reqs, NVCVAllocatorHandle halloc, NVCVTensorStorage *storage,
                 NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to tensor image batch requirements must not be NULL");
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

            static_assert(sizeof(NVCVTensorStorage) >= sizeof(priv::Tensor));
            static_assert(alignof(NVCVTensorStorage) % alignof(priv::Tensor) == 0);

            *handle = reinterpret_cast<NVCVTensorHandle>(new (storage) priv::Tensor{*reqs, alloc});
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorWrapDataConstruct,
                (const NVCVTensorData *data, NVCVTensorDataCleanupFunc cleanup, void *ctxCleanup,
                 NVCVTensorStorage *storage, NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to image batch data must not be NULL");
            }

            if (storage == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to image batch storage must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            static_assert(sizeof(NVCVTensorStorage) >= sizeof(priv::TensorWrapData));
            static_assert(alignof(NVCVTensorStorage) % alignof(priv::TensorWrapData) == 0);

            *handle
                = reinterpret_cast<NVCVTensorHandle>(new (storage) priv::TensorWrapData{*data, cleanup, ctxCleanup});
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorDestroy, (NVCVTensorHandle handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (!priv::IsDestroyed(handle))
            {
                priv::ToStaticPtr<priv::ITensor>(handle)->~ITensor();
                memset(handle, 0, sizeof(NVCVTensorStorage));

                NVCV_ASSERT(priv::IsDestroyed(handle));
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorGetFormat, (NVCVTensorHandle handle, NVCVImageFormat *fmt))
{
    return priv::ProtectCall(
        [&]
        {
            if (fmt == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image format cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            *fmt = tensor.format().value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorGetLayout, (NVCVTensorHandle handle, NVCVTensorLayout *layout))
{
    return priv::ProtectCall(
        [&]
        {
            if (layout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to tensor layout output cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            *layout = tensor.layout();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorGetAllocator, (NVCVTensorHandle handle, NVCVAllocatorHandle *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output allocator handle cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            *halloc = tensor.alloc().handle();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorExportData, (NVCVTensorHandle handle, NVCVTensorData *data))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output image batch data cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);
            tensor.exportData(*data);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorGetDimsNCHW,
                (NVCVTensorHandle handle, int32_t *batch, int32_t *channels, int32_t *height, int32_t *width))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            priv::DimsNCHW dims = tensor.dims();

            if (batch)
            {
                *batch = dims.n;
            }
            if (channels)
            {
                *channels = dims.c;
            }
            if (height)
            {
                *height = dims.h;
            }
            if (width)
            {
                *width = dims.w;
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorGetShape, (NVCVTensorHandle handle, int32_t *ndims, int32_t *shape))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            if (ndims == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input pointer to ndims cannot be NULL");
            }

            // Number of shape elements to copy
            int n = std::min(*ndims, priv::GetNDims(tensor.layout()));
            if (n > 0)
            {
                if (shape == nullptr)
                {
                    throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to shape output cannot be NULL");
                }

                NVCV_ASSERT(*ndims - n >= 0);
                std::fill_n(shape, *ndims - n, 1);
                std::copy_n(tensor.shape().end() - n, n, shape + *ndims - n);
            }

            *ndims = priv::GetNDims(tensor.layout());
        });
}
