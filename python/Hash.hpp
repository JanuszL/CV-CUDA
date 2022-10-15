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

#include <functional>
#include <tuple>
#include <type_traits>

namespace nv::cvpy {

template<class... TT>
size_t ComputeHash(const std::tuple<TT...> &a);

template<class T>
requires(!std::is_enum_v<T>) size_t ComputeHash(const T &a)
{
    return std::hash<T>{}(a);
}

template<class T>
requires(std::is_enum_v<T>) size_t ComputeHash(const T &a)
{
    using Base = typename std::underlying_type<T>::type;

    return std::hash<Base>{}(static_cast<Base>(a));
}

template<class HEAD, class... TAIL>
size_t ComputeHash(const HEAD &a, const TAIL &...aa)
{
    return ComputeHash(a) ^ (ComputeHash(aa...) << 1);
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

} // namespace nv::cvpy

#endif // NVCV_PYTHON_HASH_HPP
