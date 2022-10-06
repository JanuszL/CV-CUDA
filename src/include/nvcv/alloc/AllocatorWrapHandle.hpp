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
 * @file Allocator.hpp
 *
 * @brief Defines the public C++ implementation of allocators that wraps C allocators.
 */

#ifndef NVCV_ALLOCATOR_WRAP_HANDLE_HPP
#define NVCV_ALLOCATOR_WRAP_HANDLE_HPP

#include "../detail/CheckError.hpp"
#include "IAllocator.hpp"
#include "IResourceAllocator.hpp"

#include <cassert>
#include <functional>
#include <initializer_list>
#include <memory>

namespace nv { namespace cv {

// Used to wrap an existing NVCVAllocator into a C++ class.
// The class doesn't own the handle.
// Used when interfacing with other libraries that use NVCV C objects.
// Does the opposite of "IAllocator::handle()"
class AllocatorWrapHandle final : public virtual IAllocator
{
public:
    explicit AllocatorWrapHandle(NVCVAllocator *handle)
        : m_handle(handle)
        , m_allocHostMem(handle)
        , m_allocHostPinnedMem(handle)
        , m_allocDeviceMem(handle)
    {
    }

private:
    class HostMemAllocator final : public IHostMemAllocator
    {
    public:
        HostMemAllocator(NVCVAllocator *handle)
            : m_handle(handle)
        {
        }

    private:
        NVCVAllocator *m_handle;

        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            detail::CheckThrow(nvcvAllocatorAllocHostMemory(m_handle, &ptr, size, align));
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            nvcvAllocatorFreeHostMemory(m_handle, ptr, size, align);
        }
    };

    class HostPinnedMemAllocator final : public IHostPinnedMemAllocator
    {
    public:
        HostPinnedMemAllocator(NVCVAllocator *handle)
            : m_handle(handle)
        {
        }

    private:
        NVCVAllocator *m_handle;

        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            detail::CheckThrow(nvcvAllocatorAllocHostPinnedMemory(m_handle, &ptr, size, align));
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            nvcvAllocatorFreeHostPinnedMemory(m_handle, ptr, size, align);
        }
    };

    class DeviceMemAllocator final : public IDeviceMemAllocator
    {
    public:
        DeviceMemAllocator(NVCVAllocator *handle)
            : m_handle(handle)
        {
        }

    private:
        NVCVAllocator *m_handle;

        void *doAlloc(int64_t size, int32_t align) override
        {
            void *ptr;
            detail::CheckThrow(nvcvAllocatorAllocDeviceMemory(m_handle, &ptr, size, align));
            return ptr;
        }

        void doFree(void *ptr, int64_t size, int32_t align) noexcept override
        {
            nvcvAllocatorFreeDeviceMemory(m_handle, ptr, size, align);
        }
    };

    NVCVAllocator *m_handle;

    HostMemAllocator       m_allocHostMem;
    HostPinnedMemAllocator m_allocHostPinnedMem;
    DeviceMemAllocator     m_allocDeviceMem;

    NVCVAllocator *doGetHandle() const noexcept override
    {
        return m_handle;
    }

    IHostMemAllocator &doGetHostMemAllocator() override
    {
        return m_allocHostMem;
    }

    IHostPinnedMemAllocator &doGetHostPinnedMemAllocator() override
    {
        return m_allocHostPinnedMem;
    }

    IDeviceMemAllocator &doGetDeviceMemAllocator() override
    {
        return m_allocDeviceMem;
    }
};

}} // namespace nv::cv

#endif // NVCV_ALLOCATOR_WRAP_HANDLE_HPP
