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

#include "DefaultAllocator.hpp"

#include <cuda_runtime.h>
#include <nvcv/Version.h>
#include <util/CheckError.hpp>

#include <algorithm>
#include <cstdlib> // for aligned_alloc

namespace nv::cv::priv {

Version DefaultAllocator::doGetVersion() const
{
    return CURRENT_VERSION;
}

void *DefaultAllocator::allocHostMem(int64_t size, int32_t align)
{
    return std::aligned_alloc(align, size);
}

void DefaultAllocator::freeHostMem(void *ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;
    std::free(ptr);
}

void *DefaultAllocator::allocHostPinnedMem(int64_t size, int32_t align)
{
    void *ptr = nullptr;
    NVCV_CHECK_THROW(::cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined | cudaHostAllocMapped));
    // TODO: can we do better than this?
    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
    {
        NVCV_CHECK_LOG(::cudaFreeHost(ptr));
        throw Exception(NVCV_ERROR_INTERNAL, "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                        align);
    }
    return ptr;
}

void DefaultAllocator::freeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;
    NVCV_CHECK_LOG(::cudaFreeHost(ptr));
}

void *DefaultAllocator::allocDeviceMem(int64_t size, int32_t align)
{
    void *ptr = nullptr;
    NVCV_CHECK_THROW(::cudaMalloc(&ptr, size));

    // TODO: can we do better than this?
    if (reinterpret_cast<uintptr_t>(ptr) % align != 0)
    {
        NVCV_CHECK_LOG(::cudaFree(ptr));
        throw Exception(NVCV_ERROR_INTERNAL, "Can't allocate %ld bytes of CUDA memory with alignment at %d bytes", size,
                        align);
    }
    return ptr;
}

void DefaultAllocator::freeDeviceMem(void *ptr, int64_t size, int32_t align) noexcept
{
    (void)size;
    (void)align;
    NVCV_CHECK_LOG(::cudaFree(ptr));
}

} // namespace nv::cv::priv
