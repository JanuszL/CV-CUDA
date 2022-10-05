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

#include "IAllocator.hpp"

#include <util/Math.hpp>

namespace nv::cv::priv {

void *IAllocator::allocHostMem(int64_t size, int32_t align)
{
    if (size < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Host memory allocator size must be >= 0, not %ld", size);
    }

    if (!util::IsPowerOfTwo(align))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment when allocating host memory must be a power of two, not %d", align);
    }

    return doAllocHostMem(size, align);
}

void IAllocator::freeHostMem(void *ptr, int64_t size, int32_t align) noexcept
{
    doFreeHostMem(ptr, size, align);
}

void *IAllocator::allocHostPinnedMem(int64_t size, int32_t align)
{
    if (size < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Host-pinned memory allocator size must be >= 0, not %ld", size);
    }

    if (!util::IsPowerOfTwo(align))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment when allocating host-pinned memory must be a power of two, not %d", align);
    }

    return doAllocHostPinnedMem(size, align);
}

void IAllocator::freeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept
{
    doFreeHostPinnedMem(ptr, size, align);
}

void *IAllocator::allocDeviceMem(int64_t size, int32_t align)
{
    if (size < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Device memory allocator size must be >= 0, not %ld", size);
    }

    if (!util::IsPowerOfTwo(align))
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment when allocating device memory must be a power of two, not %d", align);
    }

    return doAllocDeviceMem(size, align);
}

void IAllocator::freeDeviceMem(void *ptr, int64_t size, int32_t align) noexcept
{
    doFreeDeviceMem(ptr, size, align);
}

} // namespace nv::cv::priv
