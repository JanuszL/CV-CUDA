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

#ifndef NVCV_CUDA_DETAIL_RANGE_CAST_IMPL_HPP
#define NVCV_CUDA_DETAIL_RANGE_CAST_IMPL_HPP

// Internal implementation of range cast functionality.
// Not to be used directly.

#include "MathWrappersImpl.hpp" // for RoundImpl, etc.
#include "Metaprogramming.hpp"  // for TypeTraits, etc.
#include "SaturateCastImpl.hpp" // for BaseSaturateImpl, etc.

namespace nv::cv::cuda::detail {

template<typename T, typename U>
inline __host__ __device__ T RangeCastImpl(U u)
{
    if constexpr (std::is_floating_point_v<U> && std::is_floating_point_v<T> && sizeof(U) > sizeof(T))
    {
        // any-float -> any-float, big -> small
        return u <= -TypeTraits<T>::max ? -TypeTraits<T>::max
                                        : (u >= TypeTraits<T>::max ? TypeTraits<T>::max : static_cast<T>(u));
    }
    else if constexpr (std::is_floating_point_v<U> && std::is_integral_v<T> && std::is_signed_v<T>)
    {
        // any-float -> any-integral-signed
        return u >= U{1} ? TypeTraits<T>::max
                         : (u <= U{-1} ? -TypeTraits<T>::max : RoundImpl<T, U>(TypeTraits<T>::max * u));
    }
    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U> && std::is_floating_point_v<T>)
    {
        // any-integral-signed -> any-float
        constexpr T invmax = T{1} / TypeTraits<U>::max;

        T out = static_cast<T>(u) * invmax;
        return out < T{-1} ? T{-1} : out;
    }
    else if constexpr (std::is_floating_point_v<U> && std::is_integral_v<T> && std::is_unsigned_v<T>)
    {
        // any-float -> any-integral-unsigned
        return u >= U{1} ? TypeTraits<T>::max : (u <= U{0} ? T{0} : RoundImpl<T, U>(TypeTraits<T>::max * u));
    }
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U> && std::is_floating_point_v<T>)
    {
        // any-integral-unsigned -> any-float
        constexpr T invmax = T{1} / TypeTraits<U>::max;
        return static_cast<T>(u) * invmax;
    }
    else if constexpr (std::is_integral_v<U> && std::is_integral_v<T>)
    {
        // any-integral -> any-integral, range cast reduces to saturate cast
        return BaseSaturateCastImpl<T, U>(u);
    }
    else
    {
        // any-float -> any-float, small -> big and equal, range cast reduces to none
        return u;
    }
}

} // namespace nv::cv::cuda::detail

#endif // NVCV_CUDA_DETAIL_RANGE_CAST_IMPL_HPP
