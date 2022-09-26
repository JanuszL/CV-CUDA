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

#ifndef NVCV_CUSTOMRESOURCEALLOCATOR_HPP
#define NVCV_CUSTOMRESOURCEALLOCATOR_HPP

/**
 * @file CustomResourceAllocator.hpp
 *
 * @brief Defines C++ implementation of custom resource allocation.
 */

#include "IResourceAllocator.hpp"

namespace nv::cv {

// Definition ------------------

class CustomHostMemAllocator final : public IHostMemAllocator
{
public:
    using Interface = IHostMemAllocator;

    using AllocFunc = std::function<Interface::AllocFunc>;
    using FreeFunc  = std::function<Interface::FreeFunc>;

    CustomHostMemAllocator(AllocFunc alloc, FreeFunc free);

private:
    AllocFunc m_alloc;
    FreeFunc  m_free;

    void *doAlloc(int64_t size, int32_t align) override;
    void  doFree(void *ptr, int64_t size, int32_t align) noexcept override;
};

class CustomHostPinnedMemAllocator final : public IHostPinnedMemAllocator
{
public:
    using Interface = IHostPinnedMemAllocator;

    using AllocFunc = std::function<Interface::AllocFunc>;
    using FreeFunc  = std::function<Interface::FreeFunc>;

    CustomHostPinnedMemAllocator(AllocFunc alloc, FreeFunc free);

private:
    AllocFunc m_alloc;
    FreeFunc  m_free;

    void *doAlloc(int64_t size, int32_t align) override;
    void  doFree(void *ptr, int64_t size, int32_t align) noexcept override;
};

class CustomDeviceMemAllocator final : public IDeviceMemAllocator
{
public:
    using Interface = IDeviceMemAllocator;

    using AllocFunc = std::function<Interface::AllocFunc>;
    using FreeFunc  = std::function<Interface::FreeFunc>;

    CustomDeviceMemAllocator(AllocFunc alloc, FreeFunc free);

private:
    AllocFunc m_alloc;
    FreeFunc  m_free;

    void *doAlloc(int64_t size, int32_t align) override;
    void  doFree(void *ptr, int64_t size, int32_t align) noexcept override;
};

// CustomHostMemAllocator implementation ------------------

CustomHostMemAllocator::CustomHostMemAllocator(AllocFunc alloc, FreeFunc free)
    : m_alloc(std::move(alloc))
    , m_free(std::move(free))
{
}

void *CustomHostMemAllocator::doAlloc(int64_t size, int32_t align)
{
    return m_alloc(size, align);
}

void CustomHostMemAllocator::doFree(void *ptr, int64_t size, int32_t align) noexcept
{
    return m_free(ptr, size, align);
}

// CustomHostPinnedMemAllocator implementation ------------------

CustomHostPinnedMemAllocator::CustomHostPinnedMemAllocator(AllocFunc alloc, FreeFunc free)
    : m_alloc(std::move(alloc))
    , m_free(std::move(free))
{
}

void *CustomHostPinnedMemAllocator::doAlloc(int64_t size, int32_t align)
{
    return m_alloc(size, align);
}

void CustomHostPinnedMemAllocator::doFree(void *ptr, int64_t size, int32_t align) noexcept
{
    return m_free(ptr, size, align);
}

// CustomDeviceMemAllocator implementation ------------------

CustomDeviceMemAllocator::CustomDeviceMemAllocator(AllocFunc alloc, FreeFunc free)
    : m_alloc(std::move(alloc))
    , m_free(std::move(free))
{
}

void *CustomDeviceMemAllocator::doAlloc(int64_t size, int32_t align)
{
    return m_alloc(size, align);
}

void CustomDeviceMemAllocator::doFree(void *ptr, int64_t size, int32_t align) noexcept
{
    return m_free(ptr, size, align);
}

} // namespace nv::cv

#endif // NVCV_CUSTOMRESOURCEALLOCATOR_HPP
