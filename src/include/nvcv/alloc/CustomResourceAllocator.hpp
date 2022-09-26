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

namespace detail {

class CustomMemAllocatorImpl : public virtual IMemAllocator
{
public:
    using Interface = IMemAllocator;

    using AllocFunc = std::function<Interface::AllocFunc>;
    using FreeFunc  = std::function<Interface::FreeFunc>;

    CustomMemAllocatorImpl(AllocFunc alloc, FreeFunc free)
        : m_alloc(std::move(alloc))
        , m_free(std::move(free))
    {
    }

private:
    AllocFunc m_alloc;
    FreeFunc  m_free;

    void *doAlloc(int64_t size, int32_t align) override
    {
        return m_alloc(size, align);
    }

    void doFree(void *ptr, int64_t size, int32_t align) noexcept override
    {
        return m_free(ptr, size, align);
    }
};

} // namespace detail

class CustomHostMemAllocator final
    : public virtual IHostMemAllocator
    , private detail::CustomMemAllocatorImpl
{
public:
    using Interface = IHostMemAllocator;

    using detail::CustomMemAllocatorImpl::CustomMemAllocatorImpl;
};

class CustomHostPinnedMemAllocator final
    : public virtual IHostPinnedMemAllocator
    , private detail::CustomMemAllocatorImpl
{
public:
    using Interface = IHostPinnedMemAllocator;

    using detail::CustomMemAllocatorImpl::CustomMemAllocatorImpl;
};

class CustomDeviceMemAllocator final
    : public virtual IDeviceMemAllocator
    , private detail::CustomMemAllocatorImpl
{
public:
    using Interface = IDeviceMemAllocator;

    using detail::CustomMemAllocatorImpl::CustomMemAllocatorImpl;
};

} // namespace nv::cv

#endif // NVCV_CUSTOMRESOURCEALLOCATOR_HPP
