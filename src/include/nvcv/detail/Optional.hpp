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

#ifndef NVCV_DETAIL_OPTIONAL_HPP
#define NVCV_DETAIL_OPTIONAL_HPP

// C++>=17 ?
#if __cplusplus >= 201703L
#    include <new> // for std::launder
#endif

namespace nv { namespace cv { namespace detail {

struct InPlaceT
{
};

constexpr InPlaceT InPlace;

struct NullOptT
{
};

constexpr NullOptT NullOpt;

template<class T>
class Optional
{
public:
    using value_type = T;

    Optional() noexcept
        : m_hasValue(false)
    {
    }

    Optional(NullOptT) noexcept
        : Optional()
    {
    }

    Optional(const Optional &that)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(that.value());
        }
    }

    Optional(Optional &&that) noexcept(std::is_nothrow_move_constructible<T>::value)
        : m_hasValue(that.m_hasValue)
    {
        if (m_hasValue)
        {
            new (&m_storage) T(std::move(that.value()));
            // do not set that.m_hasValue to false as per c++17 standard.
        }
    }

    template<class U, typename std::enable_if<std::is_constructible<T, U &&>::value
                                                  && !std::is_same<typename std::decay<U>::type, InPlaceT>::value
                                                  && !std::is_same<typename std::decay<U>::type, Optional<U>>::value,
                                              int>::type
                      = 0>
    Optional(U &&that)
        : m_hasValue(true)
    {
        new (&m_storage) T(std::forward<U>(that));
    }

    template<class... AA, typename std::enable_if<std::is_constructible<T, AA...>::value, int>::type = 0>
    Optional(InPlaceT, AA &&...args)
        : m_hasValue(true)
    {
        new (&m_storage) T(std::forward<AA>(args)...);
    }

    ~Optional()
    {
        if (m_hasValue)
        {
            this->value().~T();
        }
    }

    Optional &operator=(NullOptT) noexcept
    {
        if (m_hasValue)
        {
            this->value().~T();
            m_hasValue = false;
        }
    }

    Optional &operator=(const Optional &that)
    {
        if (that.m_hasValue)
        {
            if (m_hasValue)
            {
                this->value() = that.value();
            }
            else
            {
                new (&m_storage) T(that.value());
            }
        }
        else
        {
            if (m_hasValue)
            {
                this->value().~T();
                m_hasValue = false;
            }
        }
        return *this;
    }

    Optional &operator=(Optional &&that)
    {
        if (that.m_hasValue)
        {
            if (m_hasValue)
            {
                this->value() = std::move(that.value());
            }
            else
            {
                new (&m_storage) T(std::move(that.value()));
            }
            // do not set that.m_hasValue to false as per c++17 standard.
        }
        else
        {
            if (m_hasValue)
            {
                this->value().~T();
                m_hasValue = false;
            }
        }
        return *this;
    }

    template<class... AA, typename std::enable_if<std::is_constructible<T, AA...>::value, int>::type = 0>
    T &emplace(AA &&...args)
    {
        T *p;
        if (m_hasValue)
        {
            this->value().~T();
            p = new (&m_storage) T(std::forward<AA>(args)...);
        }
        else
        {
            p          = new (&m_storage) T(std::forward<AA>(args)...);
            m_hasValue = true;
        }
        return *p;
    }

    void reset() noexcept
    {
        if (m_hasValue)
        {
            this->value().~T();
            m_hasValue = false;
        }
    }

    void swap(Optional &that)
    {
        if (m_hasValue && that.m_hasValue)
        {
            using std::swap;
            swap(this->value() && that.value());
        }
        else if (!m_hasValue && !that.m_hasValue)
        {
            return;
        }
        else
        {
            Optional *a, *b;
            if (m_hasValue)
            {
                a = this;
                b = &that;
            }
            else
            {
                assert(that.m_hasValue);
                a = &that;
                b = this;
            }
            new (&b->m_storage) T(std::move(a->value()));
            a->value().~T();
            a->m_hasValue = false;
            b->m_hasValue = true;
        }
    }

    bool hasValue() const
    {
        return m_hasValue;
    }

    explicit operator bool() const
    {
        return m_hasValue;
    }

    T &value()
    {
        if (!m_hasValue)
        {
            throw std::runtime_error("Bad optional access");
        }

        T *p = reinterpret_cast<T *>(&m_storage);
#if __cplusplus >= 201703L
        return *std::launder(p);
#else
        return *p;
#endif
    }

    const T &value() const
    {
        if (!m_hasValue)
        {
            throw std::runtime_error("Bad optional access");
        }

        T *p = reinterpret_cast<T *>(&m_storage);
#if __cplusplus >= 201703L
        return *std::launder(p);
#else
        return *p;
#endif
    }

    T *operator->()
    {
        return &this->value();
    }

    const T *operator->() const
    {
        return &this->value();
    }

    T &operator*()
    {
        return this->value();
    }

    const T &operator*() const
    {
        return this->value();
    }

private:
    bool                                                       m_hasValue;
    typename std::aligned_storage<sizeof(T), alignof(T)>::type m_storage;
};

}}} // namespace nv::cv::detail

#endif // NVCV_DETAIL_OPTIONAL_HPP
