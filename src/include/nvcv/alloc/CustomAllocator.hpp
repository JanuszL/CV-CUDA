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
#include "../detail/IndexSequence.hpp"
#include "AllocatorWrapHandle.hpp"
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
    // Must come before m_wrap;
    NVCVAllocator m_handle;

    std::tuple<AA...> m_resAllocators;

    AllocatorWrapHandle m_wrap;

    NVCVAllocator *doCreateAllocator()
    {
        static_assert(sizeof...(AA) <= NVCV_NUM_RESOURCE_TYPES,
                      "Maximum number of resource allocators per custom allocator exceeded.");

        NVCVCustomAllocator custAllocList[sizeof...(AA)];

        doFillAllocatorList(custAllocList, detail::MakeIndexSequence<sizeof...(AA)>());

        detail::CheckThrow(nvcvAllocatorCreateCustom(custAllocList, sizeof...(AA), &m_handle));
        return &m_handle;
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

    void doFillAllocatorList(NVCVCustomAllocator *outResAlloc, detail::IndexSequence<>)
    {
        // meta-loop termination
    }

    template<size_t HEAD, size_t... TAIL>
    void doFillAllocatorList(NVCVCustomAllocator *outResAlloc, detail::IndexSequence<HEAD, TAIL...>)
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

        outResAlloc[HEAD].resType = GetResType{}(std::get<HEAD>(m_resAllocators));

        doFillAllocator(outResAlloc[HEAD], std::get<HEAD>(m_resAllocators));

        doFillAllocatorList(outResAlloc, detail::IndexSequence<TAIL...>());
    }

    NVCVAllocator *doGetHandle() const noexcept override
    {
        return m_wrap.handle();
    }

    template<class T>
    struct FindResAlloc
    {
        template<size_t... II>
        static T *Find(std::tuple<AA...> &allocs, detail::IndexSequence<II...>, T &head)
        {
            static_assert(std::is_base_of<IResourceAllocator, T>::value, "Type must represent a resource allocator");

            // Found!
            return &head;
        }

        template<size_t I>
        static T *Find(std::tuple<AA...> &allocs, detail::IndexSequence<I>, IResourceAllocator &)
        {
            // Not found.
            return nullptr;
        }

        template<size_t HEAD, size_t NECK, size_t... TAIL>
        static T *Find(std::tuple<AA...> &allocs, detail::IndexSequence<HEAD, NECK, TAIL...>, IResourceAllocator &)
        {
            // Not found yet, try the next one.
            return Find(allocs, detail::IndexSequence<NECK, TAIL...>(), std::get<NECK>(allocs));
        }
    };

    template<class T, size_t HEAD, size_t... TAIL>
    T *doGetResAllocator(detail::IndexSequence<HEAD, TAIL...>)
    {
        // Loop through all custom allocators, try to find the one that has 'T' as base.
        return FindResAlloc<T>::Find(m_resAllocators, detail::IndexSequence<HEAD, TAIL...>(),
                                     std::get<HEAD>(m_resAllocators));
    }

    template<class T>
    T *doGetResAllocator(detail::IndexSequence<>)
    {
        // No custom allocators passed, therefore...
        return nullptr; // not found
    }

    IHostMemAllocator &doGetHostMemAllocator() override
    {
        // User-customed resource allocator defined?
        if (auto *hostAlloc = doGetResAllocator<IHostMemAllocator>(detail::MakeIndexSequence<sizeof...(AA)>()))
        {
            // return it
            return *hostAlloc;
        }
        else
        {
            // or else return the default resource allocator
            return m_wrap.hostMem();
        }
    }

    IHostPinnedMemAllocator &doGetHostPinnedMemAllocator() override
    {
        if (auto *hostPinnedAlloc
            = doGetResAllocator<IHostPinnedMemAllocator>(detail::MakeIndexSequence<sizeof...(AA)>()))
        {
            return *hostPinnedAlloc;
        }
        else
        {
            return m_wrap.hostPinnedMem();
        }
    }

    IDeviceMemAllocator &doGetDeviceMemAllocator() override
    {
        if (auto *devAlloc = doGetResAllocator<IDeviceMemAllocator>(detail::MakeIndexSequence<sizeof...(AA)>()))
        {
            return *devAlloc;
        }
        else
        {
            return m_wrap.deviceMem();
        }
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
