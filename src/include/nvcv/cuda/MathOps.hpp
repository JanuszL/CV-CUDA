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
 * @file MathOps.hpp
 *
 * @brief Defines math operations.
 */

#ifndef NVCV_CUDA_MATH_OPS_HPP
#define NVCV_CUDA_MATH_OPS_HPP

#include "StaticCast.hpp" // for StaticCast, etc.
#include "TypeTraits.hpp" // for Require, etc.

#include <utility> // for std::declval, etc.

namespace nv::cv::cuda::detail {

// clang-format off

// @brief Metavariable to check if two types are compound and have the same number of components.
template<class T, class U, class = Require<HasTypeTraits<T, U>>>
constexpr bool IsSameCompound = IsCompound<T> && TypeTraits<T>::components == TypeTraits<U>::components;

// @brief Metavariable to check that at least one type is of compound type out of two types.
// If both are compound type, then it is checked that both have the same number of components.
template<typename T, typename U, class = Require<HasTypeTraits<T, U>>>
constexpr bool OneIsCompound =
    (TypeTraits<T>::components == 0 && TypeTraits<U>::components >= 1) ||
    (TypeTraits<T>::components >= 1 && TypeTraits<U>::components == 0) ||
    IsSameCompound<T, U>;

// @brief Metavariable to check if a type is of integral type.
template<typename T, class = Require<HasTypeTraits<T>>>
constexpr bool IsIntegral = std::is_integral_v<typename TypeTraits<T>::base_type>;

// @brief Metavariable to require that at least one type is of compound type out of two integral types.
// If both are compound type, then it is required that both have the same number of components.
template<typename T, typename U, class = Require<HasTypeTraits<T, U>>>
constexpr bool OneIsCompoundAndBothAreIntegral = OneIsCompound<T, U> && IsIntegral<T> && IsIntegral<U>;

// @brief Metavariable to require that a type is a CUDA compound of integral type.
template<typename T, class = Require<HasTypeTraits<T>>>
constexpr bool IsIntegralCompound = IsIntegral<T> && IsCompound<T>;

// clang-format on

} // namespace nv::cv::cuda::detail

/**
 * @brief Operators on CUDA compound types resembling the same operator on corresponding regular C type
 *
 * @details This whole group defines a set of arithmetic and bitwise operators defined on CUDA compound types.
 * They work the same way as the corresponding regular C type.  For instance, three int3 a, b and c, will accept
 * the operation a += b * c (see example below).  Furthermore, the operators accept mixed operands as CUDA compound
 * and regular C types, e.g. two int3 a and b and one int c will accept the operation a += b * c, where the scalar
 * c propagates its value for all components of b in the multiplication and the int3 result in the assignment to a.
 *
 * @defgroup NVCV_CPP_CUDATOOLS_MATHOPERATORS Math operators
 * @{
 *
 * @code
 * using PixelType = ...;
 * PixelType pix = ...;
 * float kernel = ...;
 * ConvertBaseTypeTo<float, PixelType> res = {0};
 * res += kernel * pix;
 * @endcode
 *
 * @tparam T Type of the first CUDA compound or regular C type operand
 * @tparam U Type of the second CUDA compound or regular C type operand
 *
 * @param[in] a First operand
 * @param[in] b Second operand
 *
 * @return Return value of applying the operator on \p a and \p b
 */

#define NVCV_CUDA_UNARY_OPERATOR(OPERATOR, REQUIREMENT)                                                              \
    template<typename T, class = nv::cv::cuda::Require<REQUIREMENT<T>>>                                              \
    inline __host__ __device__ auto operator OPERATOR(T a)                                                           \
    {                                                                                                                \
        using RT = nv::cv::cuda::ConvertBaseTypeTo<decltype(OPERATOR std::declval<nv::cv::cuda::BaseType<T>>()), T>; \
        if constexpr (nv::cv::cuda::NumElements<RT> == 1)                                                            \
            return RT{OPERATOR a.x};                                                                                 \
        else if constexpr (nv::cv::cuda::NumElements<RT> == 2)                                                       \
            return RT{OPERATOR a.x, OPERATOR a.y};                                                                   \
        else if constexpr (nv::cv::cuda::NumElements<RT> == 3)                                                       \
            return RT{OPERATOR a.x, OPERATOR a.y, OPERATOR a.z};                                                     \
        else if constexpr (nv::cv::cuda::NumElements<RT> == 4)                                                       \
            return RT{OPERATOR a.x, OPERATOR a.y, OPERATOR a.z, OPERATOR a.w};                                       \
    }

NVCV_CUDA_UNARY_OPERATOR(-, nv::cv::cuda::IsCompound)
NVCV_CUDA_UNARY_OPERATOR(+, nv::cv::cuda::IsCompound)
NVCV_CUDA_UNARY_OPERATOR(~, nv::cv::cuda::detail::IsIntegralCompound)

#undef NVCV_CUDA_UNARY_OPERATOR

#define NVCV_CUDA_BINARY_OPERATOR(OPERATOR, REQUIREMENT)                                                            \
    template<typename T, typename U, class = nv::cv::cuda::Require<REQUIREMENT<T, U>>>                              \
    inline __host__ __device__ auto operator OPERATOR(T a, U b)                                                     \
    {                                                                                                               \
        using RT = nv::cv::cuda::MakeType<                                                                          \
            decltype(std::declval<nv::cv::cuda::BaseType<T>>() OPERATOR std::declval<nv::cv::cuda::BaseType<U>>()), \
            nv::cv::cuda::NumComponents<T> == 0 ? nv::cv::cuda::NumComponents<U> : nv::cv::cuda::NumComponents<T>>; \
        if constexpr (nv::cv::cuda::NumComponents<T> == 0)                                                          \
        {                                                                                                           \
            if constexpr (nv::cv::cuda::NumElements<RT> == 1)                                                       \
                return RT{a OPERATOR b.x};                                                                          \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 2)                                                  \
                return RT{a OPERATOR b.x, a OPERATOR b.y};                                                          \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 3)                                                  \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z};                                          \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 4)                                                  \
                return RT{a OPERATOR b.x, a OPERATOR b.y, a OPERATOR b.z, a OPERATOR b.w};                          \
        }                                                                                                           \
        else if constexpr (nv::cv::cuda::NumComponents<U> == 0)                                                     \
        {                                                                                                           \
            if constexpr (nv::cv::cuda::NumElements<RT> == 1)                                                       \
                return RT{a.x OPERATOR b};                                                                          \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 2)                                                  \
                return RT{a.x OPERATOR b, a.y OPERATOR b};                                                          \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 3)                                                  \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b};                                          \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 4)                                                  \
                return RT{a.x OPERATOR b, a.y OPERATOR b, a.z OPERATOR b, a.w OPERATOR b};                          \
        }                                                                                                           \
        else                                                                                                        \
        {                                                                                                           \
            if constexpr (nv::cv::cuda::NumElements<RT> == 1)                                                       \
                return RT{a.x OPERATOR b.x};                                                                        \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 2)                                                  \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y};                                                      \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 3)                                                  \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z};                                    \
            else if constexpr (nv::cv::cuda::NumElements<RT> == 4)                                                  \
                return RT{a.x OPERATOR b.x, a.y OPERATOR b.y, a.z OPERATOR b.z, a.w OPERATOR b.w};                  \
        }                                                                                                           \
    }                                                                                                               \
    template<typename T, typename U, class = nv::cv::cuda::Require<nv::cv::cuda::IsCompound<T>>>                    \
    inline __host__ __device__ T &operator OPERATOR##=(T &a, U b)                                                   \
    {                                                                                                               \
        return a = nv::cv::cuda::StaticCast<nv::cv::cuda::BaseType<T>>(a OPERATOR b);                               \
    }

NVCV_CUDA_BINARY_OPERATOR(-, nv::cv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(+, nv::cv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(*, nv::cv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(/, nv::cv::cuda::detail::OneIsCompound)
NVCV_CUDA_BINARY_OPERATOR(%, nv::cv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(&, nv::cv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(|, nv::cv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(^, nv::cv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(<<, nv::cv::cuda::detail::OneIsCompoundAndBothAreIntegral)
NVCV_CUDA_BINARY_OPERATOR(>>, nv::cv::cuda::detail::OneIsCompoundAndBothAreIntegral)

#undef NVCV_CUDA_BINARY_OPERATOR

template<typename T, typename U, class = nv::cv::cuda::Require<nv::cv::cuda::detail::IsSameCompound<T, U>>>
inline __host__ __device__ bool operator==(T a, U b)
{
    if constexpr (nv::cv::cuda::NumElements<T> >= 1)
        if (a.x != b.x)
            return false;
    if constexpr (nv::cv::cuda::NumElements<T> >= 2)
        if (a.y != b.y)
            return false;
    if constexpr (nv::cv::cuda::NumElements<T> >= 3)
        if (a.z != b.z)
            return false;
    if constexpr (nv::cv::cuda::NumElements<T> == 4)
        if (a.w != b.w)
            return false;
    return true;
}

template<typename T, typename U, class = nv::cv::cuda::Require<nv::cv::cuda::detail::IsSameCompound<T, U>>>
inline __host__ __device__ bool operator!=(T a, U b)
{
    return !(a == b);
}

/**@}*/

#endif // NVCV_CUDA_MATH_OPS_HPP
