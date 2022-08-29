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

#include <nvcv/MemAllocator.h>
#include <nvcv/MemAllocator.hpp>
#include <private/core/Exception.hpp>
#include <private/core/MemAllocator.hpp>
#include <private/core/Status.hpp>
#include <util/SymbolVersioning.hpp>

#include <memory>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemAllocatorCreate,
                (const NVCVCustomMemAllocator *customAllocators, int32_t numCustomAllocators, NVCVMemAllocator *halloc))
{
    return priv::ProtectCall(
        [&]
        {
            if (halloc == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output handle must not be NULL");
            }

            auto obj = std::make_unique<priv::MemAllocator>(customAllocators, numCustomAllocators);
            *halloc  = obj->handle();
            obj.release();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemAllocatorDestroy, (NVCVMemAllocator halloc))
{
    return priv::ProtectCall([&] { delete priv::ToPtr<priv::IMemAllocator>(halloc); });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemAllocatorAllocMemory,
                (NVCVMemAllocator halloc, NVCVMemoryType memType, void **ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output buffer must not be NULL");
            }

            auto &alloc = priv::ToRef<priv::IMemAllocator>(halloc);
            *ptr        = alloc.allocMem(memType, sizeBytes, alignBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemAllocatorFreeMemory,
                (NVCVMemAllocator halloc, NVCVMemoryType memType, void *ptr, int64_t sizeBytes, int32_t alignBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (ptr != nullptr)
            {
                auto &alloc = priv::ToRef<priv::IMemAllocator>(halloc);
                alloc.freeMem(memType, ptr, sizeBytes, alignBytes);
            }
        });
}
