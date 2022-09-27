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

#include <nvcv/alloc/Allocator.h>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <private/core/CustomAllocator.hpp>
#include <private/core/DefaultAllocator.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <util/SymbolVersioning.hpp>

#include <memory>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorCreateCustom,
                (const NVCVCustomAllocator *customAllocators, int32_t numCustomAllocators, NVCVAllocator *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            std::unique_ptr<priv::IAllocator> obj;

            if (numCustomAllocators != 0)
            {
                obj = std::make_unique<priv::CustomAllocator>(customAllocators, numCustomAllocators);
            }
            else
            {
                obj = std::make_unique<priv::DefaultAllocator>();
            }

            *halloc = obj->handle();
            obj.release();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorDestroy, (NVCVAllocator halloc))
{
    return priv::ProtectCall([&] { delete priv::ToStaticPtr<priv::IAllocator>(halloc); });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorAllocHostMemory,
                (NVCVAllocator halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            auto &alloc = priv::ToStaticRef<priv::IAllocator>(halloc);
            *ptr        = alloc.allocHostMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorFreeHostMemory,
                (NVCVAllocator halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                auto &alloc = priv::ToStaticRef<priv::IAllocator>(halloc);
                alloc.freeHostMem(ptr, sizeBytes, alignBytes);
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorAllocHostPinnedMemory,
                (NVCVAllocator halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            auto &alloc = priv::ToStaticRef<priv::IAllocator>(halloc);
            *ptr        = alloc.allocHostPinnedMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorFreeHostPinnedMemory,
                (NVCVAllocator halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                auto &alloc = priv::ToStaticRef<priv::IAllocator>(halloc);
                alloc.freeHostPinnedMem(ptr, sizeBytes, alignBytes);
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorAllocDeviceMemory,
                (NVCVAllocator halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            auto &alloc = priv::ToStaticRef<priv::IAllocator>(halloc);
            *ptr        = alloc.allocDeviceMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorFreeDeviceMemory,
                (NVCVAllocator halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                auto &alloc = priv::ToStaticRef<priv::IAllocator>(halloc);
                alloc.freeDeviceMem(ptr, sizeBytes, alignBytes);
            }
        });
}
