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

#ifndef NVCV_PYTHON_HASH_HPP
#define NVCV_PYTHON_HASH_HPP

#include <nvcv/Size.hpp>

#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nv::cvpy {

template<class T>
requires(!std::is_enum_v<T> && std::is_default_constructible_v<std::hash<T>>) size_t ComputeHash(const T &a)
{
    return std::hash<T>{}(a);
}

template<class T>
requires(std::is_enum_v<T>) size_t ComputeHash(const T &a)
{
    using Base = typename std::underlying_type<T>::type;

    return std::hash<Base>{}(static_cast<Base>(a));
}

template<std::ranges::range R>
size_t ComputeHash(const R &a);

template<class... TT>
size_t ComputeHash(const std::tuple<TT...> &a);

template<class HEAD, class... TAIL>
requires(sizeof...(TAIL) >= 1) size_t ComputeHash(const HEAD &a, const TAIL &...aa)
{
    return ComputeHash(a) ^ (ComputeHash(aa...) << 1);
}

template<std::ranges::range R>
size_t ComputeHash(const R &a)
{
    size_t hash = ComputeHash(std::ranges::size(a));
    for (const auto &v : a)
    {
        hash = ComputeHash(hash, v);
    }
    return hash;
}

// Hashing for tuples ---------------------
namespace detail {
template<std::size_t... IDX, class T>
size_t ComputeHashTupleHelper(std::index_sequence<IDX...>, const T &a)
{
    return ComputeHash(std::get<IDX>(a)...);
}
} // namespace detail

template<class... TT>
size_t ComputeHash(const std::tuple<TT...> &a)
{
    return detail::ComputeHashTupleHelper(std::index_sequence_for<TT...>(), a);
}

inline size_t ComputeHash()
{
    return ComputeHash(612 /* any value works */);
}

} // namespace nv::cvpy

namespace nv::cv {

inline size_t ComputeHash(const Size2D &s)
{
    using cvpy::ComputeHash;
    return ComputeHash(s.w, s.h);
}

} // namespace nv::cv

#endif // NVCV_PYTHON_HASH_HPP
