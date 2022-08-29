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

#include "MemAllocator.hpp"

#include <cuda_runtime.h>
#include <nvcv/Version.h>
#include <util/CheckError.hpp>

#include <algorithm>
#include <cstdlib> // for aligned_alloc

namespace nv::cv::priv {

// MemAllocator ---------------------------

MemAllocator::MemAllocator(const NVCVCustomMemAllocator *customAllocators, int32_t numCustomAllocators)
{
    if (customAllocators == nullptr && numCustomAllocators != 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Array to custom allocators must not be NULL if custom allocator count is > 0");
    }

    std::ranges::fill(m_allocators, NVCVCustomMemAllocator{});

    for (int i = 0; i < numCustomAllocators; ++i)
    {
        NVCV_ASSERT(customAllocators != nullptr);

        const NVCVCustomMemAllocator &custAlloc = customAllocators[i];

        if (custAlloc.memType < 0 || custAlloc.memType > NVCV_NUM_MEMORY_TYPES)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Memory type '%d' is not understood", (int)custAlloc.memType);
        }

        if (custAlloc.fnMemAlloc == nullptr)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Custom memory allocation function for type " << custAlloc.memType << " must not be NULL";
        }
        if (custAlloc.fnMemFree == nullptr)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Custom memory deallocation function for type " << custAlloc.memType << " must not be NULL";
        }

        if (m_allocators[custAlloc.memType].fnMemAlloc != nullptr)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Custom memory allocator for type " << custAlloc.memType << " is already defined";
        }

        m_allocators[custAlloc.memType] = custAlloc;
    }

    // Now go through all allocators, find the ones that aren't customized
    // and set them to corresponding default allocator.

    for (int i = 0; i < NVCV_NUM_MEMORY_TYPES; ++i)
    {
        if (m_allocators[i].fnMemAlloc != nullptr)
        {
            NVCV_ASSERT(m_allocators[i].fnMemFree != nullptr);
            continue;
        }

        NVCV_ASSERT(m_allocators[i].fnMemFree == nullptr);

        switch (i)
        {
        case NVCV_MEM_HOST:
            static auto hostAlloc = [](void *ctx, int64_t size, int32_t align)
            {
                (void)ctx;
                return std::aligned_alloc(align, size);
            };
            static auto hostFree = [](void *ctx, void *ptr, int64_t size, int32_t align)
            {
                (void)size;
                (void)align;
                (void)ctx;
                std::free(ptr);
            };
            m_allocators[i].fnMemAlloc = hostAlloc;
            m_allocators[i].fnMemFree  = hostFree;
            break;

        case NVCV_MEM_DEVICE:
            static auto cudaAlloc = [](void *ctx, int64_t size, int32_t align) -> void *
            {
                void       *ptr = nullptr;
                cudaError_t err = ::cudaMalloc(&ptr, size);
                if (err == cudaSuccess)
                {
                    // TODO: can we do better than this?
                    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
                    {
                        NVCV_CHECK_LOG(::cudaFree(ptr));
                        throw Exception(NVCV_ERROR_INTERNAL,
                                        "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                                        align);
                    }

                    return ptr;
                }
                else
                {
                    // How about errors other than cudaErrorMemoryAllocation
                    return nullptr;
                }
            };
            static auto cudaFree = [](void *ctx, void *ptr, int64_t size, int32_t align)
            {
                (void)size;
                (void)align;
                NVCV_CHECK_LOG(::cudaFree(ptr));
            };
            m_allocators[i].fnMemAlloc = cudaAlloc;
            m_allocators[i].fnMemFree  = cudaFree;
            break;

        case NVCV_MEM_HOST_PINNED:
            static auto cudaPinnedAlloc = [](void *ctx, int64_t size, int32_t align) -> void *
            {
                void       *ptr = nullptr;
                cudaError_t err = ::cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined | cudaHostAllocMapped);
                if (err == cudaSuccess)
                {
                    // TODO: can we do better than this?
                    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
                    {
                        NVCV_CHECK_LOG(::cudaFreeHost(ptr));
                        throw Exception(NVCV_ERROR_INTERNAL,
                                        "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                                        align);
                    }

                    return ptr;
                }
                else
                {
                    // How about errors other than cudaErrorMemoryAllocation
                    return nullptr;
                }
            };
            static auto cudaPinnedFree = [](void *ctx, void *ptr, int64_t size, int32_t align)
            {
                (void)size;
                (void)align;
                NVCV_CHECK_LOG(::cudaFreeHost(ptr));
            };
            m_allocators[i].fnMemAlloc = cudaPinnedAlloc;
            m_allocators[i].fnMemFree  = cudaPinnedFree;
            break;
        }

        NVCV_ASSERT(m_allocators[i].fnMemAlloc != nullptr);
        NVCV_ASSERT(m_allocators[i].fnMemFree != nullptr);
    }
}

Version MemAllocator::doGetVersion() const
{
    return CURRENT_VERSION;
}

void *MemAllocator::allocMem(NVCVMemoryType memType, int64_t size, int32_t align)
{
    NVCV_ASSERT(0 <= memType && memType < NVCV_NUM_MEMORY_TYPES);
    NVCVCustomMemAllocator &custom = m_allocators[memType];

    NVCV_ASSERT(custom.fnMemAlloc != nullptr);
    return custom.fnMemAlloc(custom.ctx, size, align);
}

void MemAllocator::freeMem(NVCVMemoryType memType, void *ptr, int64_t size, int32_t align) noexcept
{
    NVCV_ASSERT(0 <= memType && memType < 3);
    NVCVCustomMemAllocator &custom = m_allocators[memType];

    NVCV_ASSERT(custom.fnMemFree != nullptr);
    return custom.fnMemFree(custom.ctx, ptr, size, align);
}

} // namespace nv::cv::priv
