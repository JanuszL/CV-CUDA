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

/**
 * @file IMemAllocator.hpp
 *
 * @brief Defines the public C++ interface to memory allocators.
 */

#ifndef NVCV_IMEMALLOCATOR_HPP
#define NVCV_IMEMALLOCATOR_HPP

#include "MemAllocator.h"

#include <cstddef> // for max_align_t

namespace nv { namespace cv {

enum class MemoryType
{
    HOST        = NVCV_MEM_HOST,
    HOST_PINNED = NVCV_MEM_HOST_PINNED,
    DEVICE      = NVCV_MEM_DEVICE,
};

using MemAllocFunc = void *(int64_t sizeBytes, int32_t alignBytes);
using MemFreeFunc  = void(void *ptr, int64_t sizeBytes, int32_t alignBytes);

class IMemAllocator
{
public:
    static constexpr int DEFAULT_ALIGN = alignof(std::max_align_t);

    virtual ~IMemAllocator() = default;

    NVCVMemAllocator handle() const noexcept
    {
        return doGetHandle();
    }

    void *allocMem(MemoryType memType, int64_t size, int32_t align = DEFAULT_ALIGN)
    {
        return doAllocMem(memType, size, align);
    }

    void freeMem(MemoryType memType, void *ptr, int64_t size, int32_t align = DEFAULT_ALIGN) noexcept
    {
        return doFreeMem(memType, ptr, size, align);
    }

private:
    // Using the NVI pattern.
    virtual NVCVMemAllocator doGetHandle() const noexcept = 0;

    virtual void *doAllocMem(MemoryType memType, int64_t size, int32_t align)                    = 0;
    virtual void  doFreeMem(MemoryType memType, void *ptr, int64_t size, int32_t align) noexcept = 0;
};

}} // namespace nv::cv

#endif // NVCV_IMEMALLOCATOR_HPP
