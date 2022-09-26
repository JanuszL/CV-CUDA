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
 * @brief Defines the public C++ implementation of custom allocators.
 */

#ifndef NVCV_CUSTOMALLOCATOR_HPP
#define NVCV_CUSTOMALLOCATOR_HPP

#include "../detail/CheckError.hpp"
#include "HandleWrapperAllocator.hpp"
#include "IAllocator.hpp"

#include <cassert>
#include <tuple>

namespace nv { namespace cv {

// Allows user to create custom allocators
template<class... AA>
class CustomAllocator final : public IAllocator
{
public:
    // Prohibit moves/copies.
    CustomAllocator(const CustomAllocator &) = delete;

    CustomAllocator(AA &&...allocators)
        : m_resAllocators{std::forward_as_tuple(allocators...)}
        , m_wrap{doCreateAllocator()}
    {
    }

    ~CustomAllocator()
    {
        nvcvAllocatorDestroy(m_wrap.handle());
    }

private:
    std::tuple<AA...> m_resAllocators;

    HandleWrapperAllocator m_wrap;

    NVCVAllocator doCreateAllocator()
    {
        static_assert(sizeof...(AA) <= NVCV_NUM_RESOURCE_TYPES,
                      "Maximum number of resource allocators per custom allocator exceeded.");

        NVCVCustomAllocator custAllocList[sizeof...(AA)];

        doFillAllocatorList(custAllocList, std::make_index_sequence<sizeof...(AA)>());

        NVCVAllocator halloc;
        detail::CheckThrow(nvcvAllocatorCreateCustom(custAllocList, sizeof...(AA), &halloc));
        return halloc;
    }

    void doFillAllocator(NVCVCustomAllocator &out, IMemAllocator &alloc)
    {
        static auto myMalloc = [](void *ctx_, int64_t size, int32_t align)
        {
            auto *ctx = reinterpret_cast<IMemAllocator *>(ctx_);
            assert(ctx != nullptr);

            return ctx->alloc(size, align);
        };
        static auto myFree = [](void *ctx_, void *ptr, int64_t size, int32_t align)
        {
            auto *ctx = reinterpret_cast<IMemAllocator *>(ctx_);
            assert(ctx != nullptr);

            ctx->free(ptr, size, align);
        };

        out.ctx             = &alloc;
        out.res.mem.fnAlloc = myMalloc;
        out.res.mem.fnFree  = myFree;
        // out.resType is already filled by caller
    }

    void doFillAllocatorList(NVCVCustomAllocator *outResAlloc, std::index_sequence<>)
    {
        // meta-loop termination
    }

    template<size_t HEAD, size_t... TAIL>
    void doFillAllocatorList(NVCVCustomAllocator *outResAlloc, std::index_sequence<HEAD, TAIL...>)
    {
        struct GetResType
        {
            NVCVResourceType operator()(const IHostMemAllocator &alloc) const
            {
                return NVCV_RESOURCE_MEM_HOST;
            }

            NVCVResourceType operator()(const IHostPinnedMemAllocator &alloc) const
            {
                return NVCV_RESOURCE_MEM_HOST_PINNED;
            }

            NVCVResourceType operator()(const IDeviceMemAllocator &alloc) const
            {
                return NVCV_RESOURCE_MEM_DEVICE;
            }
        };

        NVCVResourceType resType = GetResType{}(std::get<HEAD>(m_resAllocators));

        outResAlloc[HEAD].resType = resType;

        doFillAllocator(outResAlloc[HEAD], std::get<HEAD>(m_resAllocators));

        doFillAllocatorList(outResAlloc, std::index_sequence<TAIL...>());
    }

    NVCVAllocator doGetHandle() const noexcept override
    {
        return m_wrap.handle();
    }

    IHostMemAllocator &doGetHostMemAllocator() override
    {
        return m_wrap.hostMem();
    }

    IHostPinnedMemAllocator &doGetHostPinnedMemAllocator() override
    {
        return m_wrap.hostPinnedMem();
    }

    IDeviceMemAllocator &doGetDeviceMemAllocator() override
    {
        return m_wrap.deviceMem();
    }
};

// Helper function to cope with absence of CTAD (>= C++17).
template<class... AA>
CustomAllocator<AA...> CreateCustomAllocator(AA &&...allocators)
{
    return CustomAllocator(std::forward<AA>(allocators)...);
}

}} // namespace nv::cv

#endif // NVCV_CUSTOMALLOCATOR_HPP
