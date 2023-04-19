/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_ALLOC_ALLOCATOR_HPP
#define NVCV_ALLOC_ALLOCATOR_HPP

#include "../CoreResource.hpp"
#include "../detail/Callback.hpp"
#include "../detail/CompilerUtils.h"
#include "../detail/TypeTraits.hpp"
#include "Allocator.h"

#include <cassert>
#include <cstring>
#include <functional>

namespace nvcv {

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

/** A base class that encapsulates an NVCVCustomAllocator struct
 */
class ResourceAllocator
{
public:
    ResourceAllocator() = default;

    explicit ResourceAllocator(const NVCVCustomAllocator &alloc)
        : m_data(alloc)
    {
    }

    const NVCVCustomAllocator &cdata() const &
    {
        return m_data;
    }

    NVCVCustomAllocator cdata() &&
    {
        return m_data;
    }

    template<typename Derived>
    Derived cast() const
    {
        static_assert(std::is_base_of<ResourceAllocator, Derived>::value,
                      "The requested type does not inherit from ResourceAllocator");
        static_assert(std::is_constructible<Derived, NVCVCustomAllocator>::value,
                      "The requested type must be constructible from NVCVCustomAllocator");
        return Derived(m_data);
    }

    static constexpr bool IsCompatibleKind(NVCVResourceType)
    {
        return true;
    }

protected:
    NVCVCustomAllocator &data() &
    {
        return m_data;
    }

    NVCVCustomAllocator m_data{};
};

/** Encapculates a memory allocator (NVCV_RESOURCE_MEM_*)
 */
class MemAllocator : public ResourceAllocator
{
public:
    using ResourceAllocator::ResourceAllocator;

    static constexpr int DEFAULT_ALIGN = alignof(std::max_align_t);

    void *alloc(int64_t size, int32_t align = DEFAULT_ALIGN)
    {
        return m_data.res.mem.fnAlloc(m_data.ctx, size, align);
    }

    void free(void *ptr, int64_t size, int32_t align = DEFAULT_ALIGN) noexcept
    {
        m_data.res.mem.fnFree(m_data.ctx, ptr, size, align);
    }

    static constexpr bool IsCompatibleKind(NVCVResourceType resType)
    {
        return resType == NVCV_RESOURCE_MEM_HOST || resType == NVCV_RESOURCE_MEM_HOST_PINNED
            || resType == NVCV_RESOURCE_MEM_CUDA;
    }
};

namespace detail {

/** Provides a common implementation for different memory allocator wrappers
 */
template<NVCVResourceType KIND>
class MemAllocatorWithKind : public MemAllocator
{
public:
    static constexpr NVCVResourceType kResourceType = KIND;

    MemAllocatorWithKind() = default;

    MemAllocatorWithKind(const NVCVCustomAllocator &data)
        : MemAllocator(data)
    {
        if (!IsCompatibleKind(data.resType))
        {
            throw Exception(Status::ERROR_INVALID_ARGUMENT, "Incompatible allocated resource type.");
        }
    }

    static constexpr bool IsCompatibleKind(NVCVResourceType resType)
    {
        return resType == kResourceType;
    }
};

} // namespace detail

/** Encapsulates a host memory allocator descriptor
 */
class HostMemAllocator : public detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST>
{
    using Impl = detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST>;
    using Impl::Impl;
};

/** Encapsulates a host pinned memory allocator descriptor
 */
class HostPinnedMemAllocator : public detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST_PINNED>
{
    using Impl = detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_HOST_PINNED>;
    using Impl::Impl;
};

/** Encapsulates a CUDA memory allocator descriptor
 */
class CudaMemAllocator : public detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_CUDA>
{
    using Impl = detail::MemAllocatorWithKind<NVCV_RESOURCE_MEM_CUDA>;
    using Impl::Impl;
};

class Allocator : public CoreResource<NVCVAllocatorHandle, Allocator>
{
public:
    using CoreResource<NVCVAllocatorHandle, Allocator>::CoreResource;

    HostMemAllocator       hostMem() const;
    HostPinnedMemAllocator hostPinnedMem() const;
    CudaMemAllocator       cudaMem() const;

    ResourceAllocator get(NVCVResourceType resType, bool returnDefault = true) const;

    template<typename ResAlloc>
    ResAlloc get(bool returnDefault = true) const;

    virtual ~Allocator() = default;
};

inline ResourceAllocator Allocator::get(NVCVResourceType resType, bool returnDefault) const
{
    NVCVCustomAllocator data;
    detail::CheckThrow(nvcvAllocatorGet(handle(), resType, returnDefault, &data));
    return ResourceAllocator(data);
}

template<typename ResAlloc>
ResAlloc Allocator::get(bool returnDefault) const
{
    static_assert(std::is_base_of<ResourceAllocator, ResAlloc>::value,
                  "The requested resource allocator type is not derived from ResourceAllocator.");
    NVCVCustomAllocator data;
    detail::CheckThrow(nvcvAllocatorGet(handle(), ResAlloc::kResourceType, returnDefault, &data));
    return ResAlloc(data);
}

inline HostMemAllocator Allocator::hostMem() const
{
    return get<HostMemAllocator>();
}

inline HostPinnedMemAllocator Allocator::hostPinnedMem() const
{
    return get<HostPinnedMemAllocator>();
}

inline CudaMemAllocator Allocator::cudaMem() const
{
    return get<CudaMemAllocator>();
}

///////////////////////////////////////////////
// Custom allocators

template<typename AllocatorType>
class CustomMemAllocatorImpl
{
private:
    template<typename Callable>
    struct has_trivial_copy_and_destruction
        : std::integral_constant<bool, std::is_trivially_copyable<Callable>::value
                                           && std::is_trivially_destructible<Callable>::value>
    {
    };

    template<typename Callable>
    struct by_value
        : std::integral_constant<bool, has_trivial_copy_and_destruction<Callable>::value
                                           && sizeof(Callable) <= sizeof(void *)
                                           && alignof(Callable) <= alignof(void *)>
    {
    };

    template<typename T>
    static constexpr size_t DataSize()
    {
        return std::is_empty<T>::value ? 0 : sizeof(T);
    }

public:
    template<typename AllocFunction, typename FreeFunction,
             typename = detail::EnableIf_t<detail::IsInvocableR<void *, AllocFunction, int64_t, int32_t>::value>,
             typename = detail::EnableIf_t<detail::IsInvocableR<void, FreeFunction, void *, int64_t, int32_t>::value>>
    CustomMemAllocatorImpl(AllocFunction &&alloc, FreeFunction &&free)
    {
        static_assert(!std::is_lvalue_reference<AllocFunction>::value && !std::is_lvalue_reference<FreeFunction>::value,
                      "The allocation and deallocation functions must not be L-Value references. Use std::ref "
                      "if a reference is required. Note that using references will place additional requirements "
                      "on the lifetime of the function objects.");

        using T            = std::tuple<AllocFunction, FreeFunction>;
        const bool trivial = has_trivial_copy_and_destruction<AllocFunction>::value
                          && has_trivial_copy_and_destruction<FreeFunction>::value;

        const bool tuple_by_value = trivial && sizeof(T) <= sizeof(void *) && alignof(T) <= alignof(void *);

        const bool construct_from_one_value_if_equal = trivial && by_value<AllocFunction>::value
                                                    && by_value<FreeFunction>::value
                                                    && sizeof(AllocFunction) == sizeof(FreeFunction);

        // Can we fit the tuple inside a single pointer? If yes, go for it!
        if NVCV_IF_CONSTEXPR (tuple_by_value)
        {
            Construct(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                      std::integral_constant<bool, tuple_by_value>());
        }
        // Are the two callables trivial and do they context objects coincide? If yes, use that object and reinterpret the data
        // This might be useful in a common case where both alloc and free are lambdas that capture only one - and the same - pointer-like value.
        else if NVCV_IF_CONSTEXPR (construct_from_one_value_if_equal)
        {
            if (!std::memcmp(&alloc, &free, std::min(DataSize<AllocFunction>(), DataSize<FreeFunction>())))
                ConstructFromDuplicateValues(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                                             std::integral_constant<bool, construct_from_one_value_if_equal>());
            else
                Construct(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                          std::integral_constant<bool, tuple_by_value>());
        }
        // Back to square one - need to dynamically allocate the objects - we still have _one_ context object for two functions;
        // with std::function we'd end up dynamically allocating a pair of std::functions, each of which could dynamically allocate
        // - so we're still good; possibly down from 3 dynamic allocations to one
        else
        {
            Construct(std::forward<AllocFunction>(alloc), std::forward<FreeFunction>(free),
                      std::integral_constant<bool, tuple_by_value>());
        }
        m_data.resType = AllocatorType::kResourceType;
    }

    CustomMemAllocatorImpl(CustomMemAllocatorImpl &&other)
    {
        *this = std::move(other);
    }

    ~CustomMemAllocatorImpl()
    {
        reset();
    }

    bool needsCleanup() const noexcept
    {
        return m_data.cleanup != nullptr;
    }

    const NVCVCustomAllocator &cdata() const &noexcept
    {
        return m_data;
    }

    NVCVCustomAllocator release() noexcept
    {
        NVCVCustomAllocator ret = {};
        std::swap(ret, m_data);
        return ret;
    }

    void reset(NVCVCustomAllocator &&alloc) noexcept
    {
        reset();
        std::swap(m_data, alloc);
    }

    void reset() noexcept
    {
        if (m_data.cleanup)
            m_data.cleanup(m_data.ctx, &m_data);
        m_data = {};
    }

    CustomMemAllocatorImpl &operator=(CustomMemAllocatorImpl &&impl) noexcept
    {
        reset(impl.release());
        return *this;
    }

private:
    template<typename...>
    friend class CustomAllocator;

    template<typename AllocFunction, typename FreeFunction>
    void Construct(AllocFunction &&alloc, FreeFunction &&free, std::true_type)
    {
        using T = std::tuple<AllocFunction, FreeFunction>; // TODO - use something that's trivially copyable
        T ctx{std::move(alloc), std::move(free)};
        static_assert(sizeof(T) <= sizeof(void *),
                      "Internal error - this should never be invoked with a type that large.");

        m_data.res.mem.fnAlloc = [](void *c, int64_t size, int32_t align) -> void *
        {
            T     *target   = reinterpret_cast<T *>(&c);
            auto &&callable = std::get<0>(*target);
            return callable(size, align);
        };
        m_data.res.mem.fnFree = [](void *c, void *ptr, int64_t size, int32_t align)
        {
            T     *target   = reinterpret_cast<T *>(&c);
            auto &&callable = std::get<1>(*target);
            callable(ptr, size, align);
        };

        m_data.cleanup = nullptr;

        if NVCV_IF_CONSTEXPR (!std::is_empty<T>::value)
            std::memcpy(&m_data.ctx, &ctx, DataSize<T>());
    }

    template<typename AllocFunction, typename FreeFunction>
    void Construct(AllocFunction &&alloc, FreeFunction &&free, std::false_type)
    {
        using T = std::tuple<AllocFunction, FreeFunction>;
        std::unique_ptr<T> ctx(new T{std::move(alloc), std::move(free)});
        auto               cleanup = [](void *ctx, NVCVCustomAllocator *) noexcept
        {
            delete (T *)ctx;
        };

        m_data.res.mem.fnAlloc = [](void *c, int64_t size, int32_t align) -> void *
        {
            return std::get<0>(*static_cast<T *>(c))(size, align);
        };
        m_data.res.mem.fnFree = [](void *c, void *ptr, int64_t size, int32_t align)
        {
            std::get<1> (*static_cast<T *>(c))(ptr, size, align);
        };

        m_data.cleanup = cleanup;
        m_data.ctx     = ctx.release();
    }

    template<typename AllocFunction, typename FreeFunction>
    void ConstructFromDuplicateValues(AllocFunction &&alloc, FreeFunction &&free, std::true_type)
    {
        static_assert(std::is_trivially_copyable<AllocFunction>::value || std::is_empty<AllocFunction>::value,
                      "Internal error - should not pick this overload");
        static_assert(std::is_trivially_copyable<FreeFunction>::value || std::is_empty<FreeFunction>::value,
                      "Internal error - should not pick this overload");
        m_data.res.mem.fnAlloc = [](void *c, int64_t size, int32_t align) -> void *
        {
            AllocFunction *alloc = reinterpret_cast<AllocFunction *>(&c);
            return (*alloc)(size, align);
        };
        m_data.res.mem.fnFree = [](void *c, void *ptr, int64_t size, int32_t align)
        {
            FreeFunction *free = reinterpret_cast<FreeFunction *>(&c);
            (*free)(ptr, size, align);
        };

        m_data.cleanup = nullptr;
        m_data.ctx     = nullptr;
        if (DataSize<AllocFunction>() >= DataSize<FreeFunction>())
            std::memcpy(&m_data.ctx, &alloc, DataSize<AllocFunction>());
        else
            std::memcpy(&m_data.ctx, &free, DataSize<FreeFunction>());
    }

#if __cplusplus < 201703L
    template<typename AllocFunction, typename FreeFunction>
    void ConstructByOneValue(AllocFunction &&, FreeFunction &&, std::true_type, std::false_type)
    {
        assert(!"should never get here");
    }
#endif

    NVCVCustomAllocator m_data{};
};

using CustomHostMemAllocator       = CustomMemAllocatorImpl<HostMemAllocator>;
using CustomHostPinnedMemAllocator = CustomMemAllocatorImpl<HostPinnedMemAllocator>;
using CustomCudaMemAllocator       = CustomMemAllocatorImpl<CudaMemAllocator>;

template<typename... ResourceAllocators>
class CustomAllocator final : public Allocator
{
public:
    explicit CustomAllocator(ResourceAllocators &&...allocators)
    {
        NVCVCustomAllocator data[] = {allocators.cdata()...};
        NVCVAllocatorHandle h      = {};
        detail::CheckThrow(nvcvAllocatorConstructCustom(data, sizeof...(allocators), &h));
        int dummy[] = {(allocators.release(), 0)...};
        (void)dummy;
        reset(std::move(h));
    }

    ~CustomAllocator()
    {
        preDestroy();
    }

private:
    static constexpr bool kHasReferences
        = detail::Disjunction<detail::IsRefWrapper<detail::RemoveRef_t<ResourceAllocators>>...>::value;

    template<bool hasReferences = kHasReferences>
    detail::EnableIf_t<hasReferences> preDestroy()
    {
        if (this->reset() != 0)
            throw std::logic_error(
                "The allocator context contains references. The handle must not outlive the context.");
    }

    template<bool hasReferences = kHasReferences>
    detail::EnableIf_t<!hasReferences> preDestroy() noexcept
    {
    }
};

template<typename... ResourceAllocators>
CustomAllocator<ResourceAllocators...> CreateCustomAllocator(ResourceAllocators &&...allocators)
{
    return CustomAllocator<ResourceAllocators...>{std::move(allocators)...};
}

NVCV_IMPL_SHARED_HANDLE(Allocator);

} // namespace nvcv

#endif // NVCV_ALLOC_ALLOCATOR_HPP
