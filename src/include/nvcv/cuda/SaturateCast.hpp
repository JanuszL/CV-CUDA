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
 * @file SaturateCast.hpp
 *
 * @brief Defines saturate cast functionality.
 */

#ifndef NVCV_CUDA_SATURATE_CAST_HPP
#define NVCV_CUDA_SATURATE_CAST_HPP

#include "TypeTraits.hpp"              // for Require, etc.
#include "detail/SaturateCastImpl.hpp" // for SaturateCastImpl, etc.

namespace nv::cv::cuda {

/**
 * @brief Metafunction to saturate cast all elements to a target type
 *
 * @details This function saturate casts (clamping with potential rounding) all elements to the range defined by
 * the template argument type \p T.  For instance, a float4 with any values (can be below 0 and above 255) can be
 * casted to an uchar4 rounding-then-saturating each value to be in between 0 and 255 (see example below).  It is a
 * requirement of SaturateCast that both types have type traits and type \p T must be a regular C type.
 *
 * @defgroup NVCV_CPP_CUDATOOLS_SATURATECAST Saturate cast
 * @{
 *
 * @code
 * using PixelType = MakeType<uchar, 4>;
 * using FloatPixelType = ConvertBaseTypeTo<float, PixelType>;
 * FloatPixelType res = ...; // res component values are in [0, 1]
 * PixelType pix = SaturateCast<BaseType<PixelType>>(res); // pix are in [0, 255]
 * @endcode
 *
 * @tparam T Type that defines the target range to cast
 * @tparam U Type of the source value (with 1 to 4 elements) passed as argument
 *
 * @param[in] u Source value to cast all elements to range of type \p T
 *
 * @return The value with all elements clamped and potentially rounded
 */
template<typename T, typename U, class = Require<HasTypeTraits<T, U> && !IsCompound<T>>>
__host__ __device__ auto SaturateCast(U u)
{
    using RT = ConvertBaseTypeTo<T, U>;
    RT out{};

#pragma unroll
    for (int e = 0; e < NumElements<RT>; ++e)
    {
        GetElement(out, e) = detail::SaturateCastImpl<T, BaseType<U>>(GetElement(u, e));
    }

    return out;
}

/**@}*/

} // namespace nv::cv::cuda

#endif // NVCV_CUDA_SATURATE_CAST_HPP
