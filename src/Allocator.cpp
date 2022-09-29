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

namespace {

// For the most common case, when the default allocator is being used, we can
// save some cycles by directly using the singleton below.
priv::DefaultAllocator g_DefaultAllocator;
const NVCVAllocator    DEFAULT_ALLOCATOR_HANDLE = reinterpret_cast<NVCVAllocator>(&g_DefaultAllocator);

inline priv::IAllocator &GetAllocator(NVCVAllocator handle)
{
    if (handle == DEFAULT_ALLOCATOR_HANDLE)
    {
        return g_DefaultAllocator;
    }
    else
    {
        return priv::ToStaticRef<priv::IAllocator>(handle);
    }
}

} // namespace

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

            if (numCustomAllocators != 0)
            {
                auto obj = std::make_unique<priv::CustomAllocator>(customAllocators, numCustomAllocators);
                *halloc  = obj->handle();
                obj.release();
            }
            else
            {
                *halloc = DEFAULT_ALLOCATOR_HANDLE;
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvAllocatorDestroy, (NVCVAllocator halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc != DEFAULT_ALLOCATOR_HANDLE && halloc != nullptr)
            {
                delete priv::ToStaticPtr<priv::IAllocator>(halloc);
            }
        });
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

            *ptr = GetAllocator(halloc).allocHostMem(sizeBytes, alignBytes);
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
                GetAllocator(halloc).freeHostMem(ptr, sizeBytes, alignBytes);
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

            *ptr = GetAllocator(halloc).allocHostPinnedMem(sizeBytes, alignBytes);
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
                GetAllocator(halloc).freeHostPinnedMem(ptr, sizeBytes, alignBytes);
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

            *ptr = GetAllocator(halloc).allocDeviceMem(sizeBytes, alignBytes);
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
                GetAllocator(halloc).freeDeviceMem(ptr, sizeBytes, alignBytes);
            }
        });
}
