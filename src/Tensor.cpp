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
#include <private/core/AllocatorManager.hpp>
#include <private/core/Exception.hpp>
#include <private/core/IAllocator.hpp>
#include <private/core/IImage.hpp>
#include <private/core/ImageManager.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/Tensor.hpp>
#include <private/core/TensorData.hpp>
#include <private/core/TensorLayout.hpp>
#include <private/core/TensorManager.hpp>
#include <private/core/TensorWrapDataPitch.hpp>
#include <private/fmt/ImageFormat.hpp>
#include <private/fmt/PixelType.hpp>

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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorCalcRequirements,
                (int32_t ndim, const int64_t *shape, NVCVPixelType dtype, NVCVTensorLayout layout,
                 NVCVTensorRequirements *reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output requirements must not be NULL");
            }

            priv::PixelType pix{dtype};

            *reqs = priv::Tensor::CalcRequirements(ndim, shape, pix, layout);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorConstruct,
                (const NVCVTensorRequirements *reqs, NVCVAllocatorHandle halloc, NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to tensor image batch requirements must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            priv::IAllocator &alloc = priv::GetAllocator(halloc);

            *handle = priv::CreateCoreObject<priv::Tensor>(*reqs, alloc);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorWrapDataConstruct,
                (const NVCVTensorData *data, NVCVTensorDataCleanupFunc cleanup, void *ctxCleanup,
                 NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to image batch data must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            switch (data->bufferType)
            {
            case NVCV_TENSOR_BUFFER_PITCH_DEVICE:
                *handle = priv::CreateCoreObject<priv::TensorWrapDataPitch>(*data, cleanup, ctxCleanup);
                break;

            default:
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image buffer type not supported";
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorWrapImageConstruct, (NVCVImageHandle himg, NVCVTensorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (himg == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Image handle must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            auto &img = priv::ToStaticRef<priv::IImage>(himg);

            NVCVTensorData tensorData;
            FillTensorData(img, tensorData);

            *handle = priv::CreateCoreObject<priv::TensorWrapDataPitch>(tensorData, nullptr, nullptr);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorDestroy, (NVCVTensorHandle handle))
{
    return priv::ProtectCall([&] { priv::DestroyCoreObject(handle); });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetLayout, (NVCVTensorHandle handle, NVCVTensorLayout *layout))
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetAllocator, (NVCVTensorHandle handle, NVCVAllocatorHandle *halloc))
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorExportData, (NVCVTensorHandle handle, NVCVTensorData *data))
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetShape, (NVCVTensorHandle handle, int32_t *ndim, int64_t *shape))
{
    return priv::ProtectCall(
        [&]
        {
            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);

            if (ndim == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Input pointer to ndim cannot be NULL");
            }

            if (shape != nullptr)
            {
                // Number of shape elements to copy
                int n = std::min(*ndim, tensor.ndim());
                if (n > 0)
                {
                    if (shape == nullptr)
                    {
                        throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to shape output cannot be NULL");
                    }

                    NVCV_ASSERT(*ndim - n >= 0);
                    std::fill_n(shape, *ndim - n, 1);
                    std::copy_n(tensor.shape() + tensor.ndim() - n, n, shape + *ndim - n);
                }
            }

            *ndim = tensor.ndim();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvTensorGetDataType, (NVCVTensorHandle handle, NVCVPixelType *dtype))
{
    return priv::ProtectCall(
        [&]
        {
            if (dtype == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type output cannot be NULL");
            }

            auto &tensor = priv::ToStaticRef<const priv::ITensor>(handle);
            *dtype       = tensor.dtype().value();
        });
}
