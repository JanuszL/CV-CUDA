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
 * @file MemAllocator.hpp
 *
 * @brief Defines the public C++ implementation of memory allocators.
 */

#ifndef NVCV_MEMALLOCATOR_HPP
#define NVCV_MEMALLOCATOR_HPP

#include "IMemAllocator.hpp"
#include "detail/CheckError.hpp"

#include <cassert>
#include <functional>
#include <initializer_list>
#include <memory>

namespace nv { namespace cv {

class MemAllocatorWrapHandle final : public IMemAllocator
{
public:
    explicit MemAllocatorWrapHandle(NVCVMemAllocator halloc);

private:
    NVCVMemAllocator m_handle;

    NVCVMemAllocator doGetHandle() const noexcept override;

    void *doAllocMem(MemoryType memType, int64_t size, int32_t align) override;
    void  doFreeMem(MemoryType memType, void *ptr, int64_t size, int32_t align) noexcept override;
};

struct CustomMemAllocator
{
    MemoryType                  memType;
    std::function<MemAllocFunc> fnMemAlloc;
    std::function<MemFreeFunc>  fnMemFree;
};

class MemAllocator final : public IMemAllocator
{
public:
    // Prohibit moves/copies.
    MemAllocator(const MemAllocator &) = delete;

    explicit MemAllocator(std::initializer_list<CustomMemAllocator> customAllocators);
    ~MemAllocator();

private:
    struct Context
    {
        std::function<MemAllocFunc> fnMemMalloc;
        std::function<MemFreeFunc>  fnMemFree;
    };

    // Must come before m_wrap
    Context m_ctx[NVCV_NUM_MEMORY_TYPES];

    MemAllocatorWrapHandle m_wrap;

    NVCVMemAllocator doCreateAllocator(std::initializer_list<CustomMemAllocator> customAllocators);

    NVCVMemAllocator doGetHandle() const noexcept override;

    void *doAllocMem(MemoryType memType, int64_t size, int32_t align) override;
    void  doFreeMem(MemoryType memType, void *ptr, int64_t size, int32_t align) noexcept override;
};

// MemAllocatorWrapHandle implementation --------------------------------

inline MemAllocatorWrapHandle::MemAllocatorWrapHandle(NVCVMemAllocator halloc)
    : m_handle(halloc)
{
}

inline NVCVMemAllocator MemAllocatorWrapHandle::doGetHandle() const noexcept
{
    return m_handle;
}

inline void *MemAllocatorWrapHandle::doAllocMem(MemoryType memType, int64_t size, int32_t align)
{
    void *ptr;
    detail::CheckThrow(nvcvMemAllocatorAllocMemory(m_handle, static_cast<NVCVMemoryType>(memType), &ptr, size, align));
    return ptr;
}

inline void MemAllocatorWrapHandle::doFreeMem(MemoryType memType, void *ptr, int64_t size, int32_t align) noexcept
{
    nvcvMemAllocatorFreeMemory(m_handle, static_cast<NVCVMemoryType>(memType), ptr, size, align);
}

// MemAllocator implementation -----------------------

inline MemAllocator::MemAllocator(std::initializer_list<CustomMemAllocator> customAllocators)
    : m_wrap(doCreateAllocator(std::move(customAllocators)))
{
}

inline NVCVMemAllocator MemAllocator::doCreateAllocator(std::initializer_list<CustomMemAllocator> customAllocators)
{
    assert(customAllocators.size() <= NVCV_NUM_MEMORY_TYPES);

    NVCVCustomMemAllocator custAllocList[NVCV_NUM_MEMORY_TYPES];

    for (const CustomMemAllocator &custAlloc : customAllocators)
    {
        auto cMemType = static_cast<NVCVMemoryType>(custAlloc.memType);

        assert(0 <= cMemType && cMemType < NVCV_NUM_MEMORY_TYPES && "Unexpected memory type");

        static auto myMalloc = [](void *ctx_, int64_t size, int32_t align)
        {
            auto *ctx = reinterpret_cast<Context *>(ctx_);
            assert(ctx != nullptr);

            assert(ctx->fnMemMalloc);
            return ctx->fnMemMalloc(size, align);
        };
        static auto myFree = [](void *ctx_, void *ptr, int64_t size, int32_t align)
        {
            auto *ctx = reinterpret_cast<Context *>(ctx_);
            assert(ctx != nullptr);

            assert(ctx->fnMemFree);
            ctx->fnMemFree(ptr, size, align);
        };

        // Get the context used for the current memory type
        Context &ctx = m_ctx[cMemType];

        // Unfortunately we can't move stuff from an initializer_list...
        // we have to copy the functors.
        ctx.fnMemMalloc = custAlloc.fnMemAlloc;
        ctx.fnMemFree   = custAlloc.fnMemFree;

        custAllocList[cMemType].memType    = cMemType;
        custAllocList[cMemType].fnMemAlloc = myMalloc;
        custAllocList[cMemType].fnMemFree  = myFree;
        custAllocList[cMemType].ctx        = &ctx;
    }

    NVCVMemAllocator halloc;
    detail::CheckThrow(nvcvMemAllocatorCreate(custAllocList, customAllocators.size(), &halloc));
    return halloc;
}

inline MemAllocator::~MemAllocator()
{
    nvcvMemAllocatorDestroy(m_wrap.handle());
}

inline NVCVMemAllocator MemAllocator::doGetHandle() const noexcept
{
    return m_wrap.handle();
}

inline void *MemAllocator::doAllocMem(MemoryType memType, int64_t size, int32_t align)
{
    return m_wrap.allocMem(memType, size, align);
}

inline void MemAllocator::doFreeMem(MemoryType memType, void *ptr, int64_t size, int32_t align) noexcept
{
    return m_wrap.freeMem(memType, ptr, size, align);
}

}} // namespace nv::cv

#endif // NVCV_MEMALLOCATOR_HPP
