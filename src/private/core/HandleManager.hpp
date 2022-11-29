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

#ifndef NVCV_PRIV_CORE_HANDLE_MANAGER_HPP
#define NVCV_PRIV_CORE_HANDLE_MANAGER_HPP

#include <util/Algorithm.hpp>
#include <util/Assert.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace nv::cv::priv {

namespace detail {
template<class, class = void>
struct GetHandleType
{
    using type = void *;
};

template<class T>
struct GetHandleType<T, std::void_t<decltype(sizeof(typename T::HandleType))>>
{
    using type = typename T::HandleType;
};
} // namespace detail

template<class T>
using GetHandleType = typename detail::GetHandleType<T>::type;

// Must be a power of two
// We use the 4 LSB of the Resource address to store the handle generation.
// For that to work, Resource object address must be aligned to 16 bytes.
static constexpr int kResourceAlignment = 16;

template<typename Interface, typename Storage>
class HandleManager
{
    struct alignas(kResourceAlignment) Resource
    {
        // We allow the resource to be reused up to 16 times,
        // the corresponding handle will have a different value each time.
        // After that, a handle to an object that was already destroyed might
        // refer to a different object.
        uint8_t generation : 4; // must be log2(kResourceAlignment)

        Resource *next = nullptr;

        Resource();
        ~Resource();

        template<class T, typename... Args>
        T *constructObject(Args &&...args)
        {
            static_assert(std::is_base_of_v<Interface, T>);

            static_assert(sizeof(Storage) >= sizeof(T));
            static_assert(alignof(Storage) % alignof(T) == 0);

            NVCV_ASSERT(!this->live());
            T *obj   = new (m_storage) T{std::forward<Args>(args)...};
            m_ptrObj = obj;
            this->generation++;

            NVCV_ASSERT(this->live());

            return obj;
        }

        void destroyObject();

        Interface *obj() const
        {
            return m_ptrObj;
        }

        bool live() const
        {
            return m_ptrObj != nullptr;
        }

    private:
        Interface *m_ptrObj = nullptr;
        alignas(Storage) std::byte m_storage[sizeof(Storage)];
    };

public:
    using HandleType = GetHandleType<Interface>;

    HandleManager(const char *name);
    ~HandleManager();

    template<class T, typename... Args>
    HandleType create(Args &&...args)
    {
        Resource *res = doFetchFreeResource();
        res->template constructObject<T>(std::forward<Args>(args)...);
        return doGetHandleFromResource(res);
    }

    // true if handle is destroyed, false if handle is invalid (or already removed)
    bool destroy(HandleType handle);

    Interface *validate(HandleType handle) const;

    void setFixedSize(int32_t maxSize);
    void setDynamicSize(int32_t minSize = 0);

    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    void       doAllocate(size_t count);
    void       doGrow();
    Resource  *doFetchFreeResource();
    void       doReturnResource(Resource *r);
    HandleType doGetHandleFromResource(Resource *r) const;
    Resource  *doGetResourceFromHandle(HandleType handle) const;
    bool       isManagedResource(Resource *r) const;
};

template<class... AA>
struct alignas(util::Max(alignof(AA)...)) CompatibleStorage
{
    std::byte storage[util::Max(sizeof(AA)...)];
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_HANDLE_MANAGER_HPP
