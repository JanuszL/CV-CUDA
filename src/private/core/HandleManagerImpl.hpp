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

#ifndef NVCV_PRIV_CORE_HANDLE_MANAGER_IMPL_HPP
#define NVCV_PRIV_CORE_HANDLE_MANAGER_IMPL_HPP

#include "Exception.hpp"
#include "LockFreeStack.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>

namespace nv::cv::priv {

static const char *LEAK_DETECTION_ENVVAR = "NVCV_LEAK_DETECTION";

template<typename Node>
class ManagedLockFreeStack : protected LockFreeStack<Node>
{
    using Base = LockFreeStack<Node>;

public:
    using Base::pop;
    using Base::push;
    using Base::pushStack;
    using Base::top;

    template<typename... Args>
    Node *emplace(Args &&...args)
    {
        Node *n = new Node{std::forward<Args>(args)...};
        push(n);
        return n;
    }

    ~ManagedLockFreeStack()
    {
        clear();
    }

    void clear()
    {
        if (Node *h = this->release())
        {
            while (h)
            {
                auto *n = h->next;
                delete h;
                h = n;
            }
        }
    }
};

template<typename Interface, typename Storage>
HandleManager<Interface, Storage>::Resource::Resource()
{
    this->generation = 0;
}

template<typename Interface, typename Storage>
HandleManager<Interface, Storage>::Resource::~Resource()
{
    this->destroyObject();
}

template<typename Interface, typename Storage>
void HandleManager<Interface, Storage>::Resource::destroyObject()
{
    if (m_ptrObj)
    {
        m_ptrObj->~Interface();
        m_ptrObj = nullptr;
    }

    NVCV_ASSERT(!this->live());
}

template<typename Interface, typename Storage>
struct HandleManager<Interface, Storage>::Impl
{
    static constexpr int kMinHandles = 1024;

    std::mutex mtxAlloc;

    struct ResourcePool
    {
        explicit ResourcePool(size_t count)
            : resources(count)
        {
        }

        ResourcePool         *next = nullptr;
        std::vector<Resource> resources;
    };

    // Store the resources' buffer.
    // For efficient validation, it's divided into several pools,
    // it's easy to check if a given resource belong to a pool, O(1).
    ManagedLockFreeStack<ResourcePool> resourceStack;

    // All the free resources we have
    LockFreeStack<Resource> freeResources;

    static_assert(std::atomic<Resource *>::is_always_lock_free);

    bool            hasFixedSize  = false;
    int             totalCapacity = 0;
    std::atomic_int usedCount     = 0;
    const char     *name;
};

template<typename Interface, typename Storage>
HandleManager<Interface, Storage>::HandleManager(const char *name)
    : pimpl(std::make_unique<Impl>())
{
    pimpl->name = name;
}

template<typename Interface, typename Storage>
HandleManager<Interface, Storage>::~HandleManager()
{
    this->clear();
}

template<typename Interface, typename Storage>
bool HandleManager<Interface, Storage>::destroy(HandleType handle)
{
    if (this->validate(handle))
    {
        Resource *res = doGetResourceFromHandle(handle);
        res->destroyObject();

        doReturnResource(res);
        return true;
    }
    else
    {
        return false;
    }
}

template<typename Interface, typename Storage>
bool HandleManager<Interface, Storage>::isManagedResource(Resource *r) const
{
    for (auto *range = pimpl->resourceStack.top(); range; range = range->next)
    {
        auto *b = range->resources.data();
        auto *e = b + range->resources.size();
        if (r >= b && r < e)
        {
            return true;
        }
    }
    return false;
}

template<typename Interface, typename Storage>
Interface *HandleManager<Interface, Storage>::validate(HandleType handle) const
{
    if (!handle)
    {
        return nullptr;
    }

    Resource *res = doGetResourceFromHandle(handle);

    if (this->isManagedResource(res) && res->live()
        && res->generation == ((uintptr_t)handle & (kResourceAlignment - 1)))
    {
        return res->obj();
    }
    else
    {
        return nullptr;
    }
}

template<typename Interface, typename Storage>
void HandleManager<Interface, Storage>::setFixedSize(int32_t maxSize)
{
    std::lock_guard lock(pimpl->mtxAlloc);
    if (pimpl->usedCount > 0)
    {
        throw Exception(NVCV_ERROR_INVALID_OPERATION,
                        "Cannot change the size policy while there are still %d live %s handles", (int)pimpl->usedCount,
                        pimpl->name);
    }

    if (pimpl->totalCapacity >= maxSize)
    {
        return;
    }

    this->clear();

    pimpl->hasFixedSize = true;
    doAllocate(maxSize - pimpl->totalCapacity);
}

template<typename Interface, typename Storage>
void HandleManager<Interface, Storage>::setDynamicSize(int32_t minSize)
{
    std::lock_guard lock(pimpl->mtxAlloc);

    pimpl->hasFixedSize = false;
    if (pimpl->totalCapacity < minSize)
    {
        doAllocate(minSize);
    }
}

template<typename Interface, typename Storage>
void HandleManager<Interface, Storage>::clear()
{
    if (pimpl->usedCount > 0)
    {
        const char *leakDetection = getenv(LEAK_DETECTION_ENVVAR);
#ifndef NDEBUG
        // On debug builds, report leaks by default
        if (leakDetection == nullptr)
        {
            leakDetection = "warn";
        }
#endif

        if (leakDetection != nullptr)
        {
            bool doAbort = false;
            if (strcmp(leakDetection, "warn") == 0)
            {
                std::cerr << "WARNING: ";
            }
            else if (strcmp(leakDetection, "abort") == 0)
            {
                std::cerr << "ERROR: ";
                doAbort = true;
            }
            else
            {
                std::cerr << "Invalid value '" << leakDetection << " for " << LEAK_DETECTION_ENVVAR
                          << " environment variable. It must be either not defiled or '0' "
                             "(to disable), 'warn' or 'abort'";
                abort();
            }

            std::cerr << pimpl->name << " leak detection: " << pimpl->usedCount << " handle"
                      << (pimpl->usedCount > 1 ? "s" : "") << " still in use" << std::endl;
            if (doAbort)
            {
                abort();
            }
        }
    }

    pimpl->freeResources.clear();
    pimpl->resourceStack.clear();
}

template<typename Interface, typename Storage>
void HandleManager<Interface, Storage>::doAllocate(size_t count)
{
    NVCV_ASSERT(count > 0);
    auto     *res_block = pimpl->resourceStack.emplace(count);
    Resource *data      = res_block->resources.data();

    // Turn the newly allocated resources into a forward_list
    for (size_t i = 0; i < count - 1; i++)
    {
        data[i].next = &data[i + 1];
    }

    // Add them to the list of free resources.
    pimpl->freeResources.pushStack(data, data + count - 1);
}

template<typename Interface, typename Storage>
void HandleManager<Interface, Storage>::doGrow()
{
    if (pimpl->hasFixedSize)
    {
        throw Exception(NVCV_ERROR_OUT_OF_MEMORY, "%s handle manager pool exhausted under fixed size policy",
                        pimpl->name);
    }

    std::lock_guard lock(pimpl->mtxAlloc);
    if (!pimpl->freeResources.top())
    {
        doAllocate(pimpl->totalCapacity ? pimpl->totalCapacity : pimpl->kMinHandles);
    }
}

template<typename Interface, typename Storage>
auto HandleManager<Interface, Storage>::doFetchFreeResource() -> Resource *
{
    for (;;)
    {
        if (auto *r = pimpl->freeResources.pop())
        {
            pimpl->usedCount++;
            return r;
        }
        else
        {
            doGrow();
        }
    }
}

template<typename Interface, typename Storage>
void HandleManager<Interface, Storage>::doReturnResource(Resource *r)
{
    pimpl->freeResources.push(r);
    pimpl->usedCount--;
}

template<typename Interface, typename Storage>
auto HandleManager<Interface, Storage>::doGetHandleFromResource(Resource *r) const -> HandleType
{
    if (r)
    {
        return reinterpret_cast<HandleType>((uintptr_t)r | r->generation);
    }
    else
    {
        return {};
    }
}

template<typename Interface, typename Storage>
auto HandleManager<Interface, Storage>::doGetResourceFromHandle(HandleType handle) const -> Resource *
{
    if (handle)
    {
        return reinterpret_cast<Resource *>((uintptr_t)handle & -kResourceAlignment);
    }
    else
    {
        return nullptr;
    }
}

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_HANDLE_MANAGER_IMPL_HPP
