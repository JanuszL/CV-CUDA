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

#ifndef NVCV_UTIL_MATH_HPP
#define NVCV_UTIL_MATH_HPP

#include "Compiler.hpp"
#include "Metaprogramming.hpp"

#include <cassert>
#include <type_traits>

namespace nv::cv::util {

template<class T, class U, class = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<U>>>
NVCV_CUDA_HOST_DEVICE constexpr T RoundUp(T value, U multiple)
{
    return (value + multiple - 1) / multiple * multiple;
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
NVCV_CUDA_HOST_DEVICE constexpr bool IsPowerOfTwo(T value)
{
    return (value & (value - 1)) == 0;
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
NVCV_CUDA_HOST_DEVICE constexpr auto RoundUpNextPowerOfTwo(T x)
{
    assert(x >= 0);

    // Source: Hacker's Delight 1st ed, p.48,
    // adapted for any integer size.

    x = x - 1;
    // all constants, compiler can unroll it
    for (size_t i = 1; i < sizeof(T) * 8; i <<= 1)
    {
        x = x | (x >> i);
    }

    if constexpr (std::is_same_v<T, int32_t>)
    {
        return (int64_t)x + 1;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        return (uint64_t)x + 1;
    }
    else
    {
        return x + 1;
    }
}

template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
NVCV_CUDA_HOST_DEVICE constexpr auto DivUp(T num, TypeIdentity<T> den)
{
    assert(num >= 0);
    assert(den > 0);

    return (num + (den - 1)) / den;
}

} // namespace nv::cv::util

#endif // NVCV_UTIL_MATH_HPP
