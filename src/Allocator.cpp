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
#include <private/core/AllocatorManager.hpp>
#include <private/core/CustomAllocator.hpp>
#include <private/core/DefaultAllocator.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <util/Assert.h>

#include <memory>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorConstructCustom,
                (const NVCVCustomAllocator *customAllocators, int32_t numCustomAllocators, NVCVAllocatorHandle *handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            if (numCustomAllocators != 0)
            {
                *handle = priv::CreateCoreObject<priv::CustomAllocator>(customAllocators, numCustomAllocators);
            }
            else
            {
                *handle = priv::CreateCoreObject<priv::DefaultAllocator>();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorDestroy, (NVCVAllocatorHandle halloc))
{
    return priv::ProtectCall([&] { priv::DestroyCoreObject(halloc); });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorAllocHostMemory,
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorFreeHostMemory,
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorAllocHostPinnedMemory,
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorFreeHostPinnedMemory,
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorAllocDeviceMemory,
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

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvAllocatorFreeDeviceMemory,
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
