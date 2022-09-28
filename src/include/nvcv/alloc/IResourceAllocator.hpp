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

#ifndef NVCV_IRESOURCEALLOCATOR_HPP
#define NVCV_IRESOURCEALLOCATOR_HPP

#include <cassert>
#include <cstddef> // for std::max_align_t

/**
 * @file IResourceAllocator.hpp
 *
 * @brief Defines C++ interface for resource allocation.
 */

namespace nv { namespace cv {

class IResourceAllocator
{
public:
    virtual ~IResourceAllocator() = 0;
};

inline IResourceAllocator::~IResourceAllocator() {}

class IMemAllocator : public IResourceAllocator
{
public:
    static constexpr int DEFAULT_ALIGN = alignof(std::max_align_t);

    using AllocFunc = void *(int64_t size, int32_t align);
    using FreeFunc  = void(void *ptr, int64_t size, int32_t align);

    void *alloc(int64_t size, int32_t align = DEFAULT_ALIGN);
    void  free(void *ptr, int64_t size, int32_t align = DEFAULT_ALIGN) noexcept;

private:
    // NVI pattern
    virtual void *doAlloc(int64_t size, int32_t align)                    = 0;
    virtual void  doFree(void *ptr, int64_t size, int32_t align) noexcept = 0;
};

class IHostMemAllocator : public virtual IMemAllocator
{
};

class IHostPinnedMemAllocator : public virtual IMemAllocator
{
};

class IDeviceMemAllocator : public virtual IMemAllocator
{
};

// Implementation ----------------------

void *IMemAllocator::alloc(int64_t size, int32_t align)
{
    void *ptr = doAlloc(size, align);
    assert(ptr != nullptr && "nv::cv::IMemAllocator::alloc post-condition failed");
    return ptr;
}

void IMemAllocator::free(void *ptr, int64_t size, int32_t align) noexcept
{
    doFree(ptr, size, align);
}

}} // namespace nv::cv

#endif // NVCV_IRESOURCEALLOCATOR_HPP
