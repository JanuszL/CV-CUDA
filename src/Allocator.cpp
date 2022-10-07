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
#include <private/core/SymbolVersioning.hpp>
#include <util/Assert.h>

#include <memory>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorConstructCustom,
                (const NVCVCustomAllocator *customAllocators, int32_t numCustomAllocators,
                 NVCVAllocatorStorage *storage, NVCVAllocatorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (storage == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to allocator storage must not be NULL");
            }

            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            if (numCustomAllocators != 0)
            {
                static_assert(sizeof(NVCVAllocatorStorage) >= sizeof(priv::CustomAllocator));
                static_assert(alignof(NVCVAllocatorStorage) % alignof(priv::CustomAllocator) == 0);

                *handle = reinterpret_cast<NVCVAllocatorHandle>(
                    new (storage) priv::CustomAllocator{customAllocators, numCustomAllocators});
            }
            else
            {
                static_assert(sizeof(NVCVAllocatorStorage) >= sizeof(priv::DefaultAllocator));
                static_assert(alignof(NVCVAllocatorStorage) % alignof(priv::DefaultAllocator) == 0);

                *handle = reinterpret_cast<NVCVAllocatorHandle>(new (storage) priv::DefaultAllocator{});
            }

            NVCV_ASSERT(!priv::IsDestroyed(handle));
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorDestroy, (NVCVAllocatorHandle halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (!priv::IsDestroyed(halloc))
            {
                priv::ToStaticPtr<priv::IAllocator>(halloc)->~IAllocator();
                memset(halloc, 0, sizeof(NVCVAllocatorStorage));

                NVCV_ASSERT(priv::IsDestroyed(halloc));
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorAllocHostMemory,
                (NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            *ptr = priv::ToStaticRef<priv::IAllocator>(halloc).allocHostMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorFreeHostMemory,
                (NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                priv::ToStaticRef<priv::IAllocator>(halloc).freeHostMem(ptr, sizeBytes, alignBytes);
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorAllocHostPinnedMemory,
                (NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            *ptr = priv::ToStaticRef<priv::IAllocator>(halloc).allocHostPinnedMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorFreeHostPinnedMemory,
                (NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                priv::ToStaticRef<priv::IAllocator>(halloc).freeHostPinnedMem(ptr, sizeBytes, alignBytes);
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorAllocDeviceMemory,
                (NVCVAllocatorHandle halloc, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            *ptr = priv::ToStaticRef<priv::IAllocator>(halloc).allocDeviceMem(sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorFreeDeviceMemory,
                (NVCVAllocatorHandle halloc, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                priv::ToStaticRef<priv::IAllocator>(halloc).freeDeviceMem(ptr, sizeBytes, alignBytes);
            }
        });
}
