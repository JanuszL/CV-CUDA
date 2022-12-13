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
 * @file IAllocator.hpp
 *
 * @brief Defines the public C++ interface to allocators.
 */

#ifndef NVCV_ALLOC_IALLOCATOR_HPP
#define NVCV_ALLOC_IALLOCATOR_HPP

#include <cstdint>

#include "Fwd.hpp" // for NVCVAllocator

namespace nv { namespace cv {

// Helper class to explicitly assign
// address alignments.
class MemAlignment
{
public:
    MemAlignment() = default;

    int32_t baseAddr() const
    {
        return m_baseAddrAlignment;
    }

    int32_t rowAddr() const
    {
        return m_rowAddrAlignment;
    }

    MemAlignment &baseAddr(int32_t alignment)
    {
        m_baseAddrAlignment = alignment;
        return *this;
    }

    MemAlignment &rowAddr(int32_t alignment)
    {
        m_rowAddrAlignment = alignment;
        return *this;
    }

private:
    int32_t m_baseAddrAlignment = 0;
    int32_t m_rowAddrAlignment  = 0;
};

class IAllocator
{
public:
    virtual ~IAllocator() = default;

    NVCVAllocatorHandle handle() const noexcept;

    IHostMemAllocator       &hostMem();
    IHostPinnedMemAllocator &hostPinnedMem();
    IDeviceMemAllocator     &deviceMem();

private:
    // Using the NVI pattern.
    virtual NVCVAllocatorHandle doGetHandle() const noexcept = 0;

    virtual IHostMemAllocator       &doGetHostMemAllocator()       = 0;
    virtual IHostPinnedMemAllocator &doGetHostPinnedMemAllocator() = 0;
    virtual IDeviceMemAllocator     &doGetDeviceMemAllocator()     = 0;
};

inline NVCVAllocatorHandle IAllocator::handle() const noexcept
{
    return doGetHandle();
}

inline IHostMemAllocator &IAllocator::hostMem()
{
    return doGetHostMemAllocator();
}

inline IHostPinnedMemAllocator &IAllocator::hostPinnedMem()
{
    return doGetHostPinnedMemAllocator();
}

inline IDeviceMemAllocator &IAllocator::deviceMem()
{
    return doGetDeviceMemAllocator();
}

}} // namespace nv::cv

#endif // NVCV_ALLOC_IMEMALLOCATOR_HPP
