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
 * @file DropCast.hpp
 *
 * @brief Defines drop cast functionality.
 */

#ifndef NVCV_CUDA_DROP_CAST_HPP
#define NVCV_CUDA_DROP_CAST_HPP

#include "TypeTraits.hpp" // for Require, etc.

namespace nv::cv::cuda {

/**
 * @brief Metafunction to drop components of a compound value
 *
 * @details The template parameter \p N defines the number of components to cast the CUDA compound type \p T passed
 * as function argument \p v.  This is done by dropping the last components after \p N from \p v.  For instance, an
 * uint3 can have its z component dropped by passing it as function argument to DropCast and the number 2 as
 * template argument (see example below).  The type \p T is not needed as it is inferred from the argument \p v.
 * It is a requirement of the DropCast function that the type \p T has at least N components.
 *
 * @defgroup NVCV_CPP_CUDATOOLS_DROPCAST Drop Cast
 * @{
 *
 * @code
 * uint2 dstIdx = DropCast<2>(blockIdx * blockDim + threadIdx);
 * @endcode
 *
 * @tparam N Number of components to return
 *
 * @param[in] v Value to drop components from
 *
 * @return The compound value with N components dropping the last, extra components
 */
template<int N, typename T, class = Require<HasEnoughComponents<T, N>>>
__host__ __device__ auto DropCast(T v)
{
    using RT = MakeType<BaseType<T>, N>;
    if constexpr (std::is_same_v<T, RT>)
    {
        return v;
    }
    else
    {
        RT out{};

#pragma unroll
        for (int e = 0; e < NumElements<RT>; ++e)
        {
            GetElement(out, e) = GetElement(v, e);
        }

        return out;
    }
}

/**@}*/

} // namespace nv::cv::cuda

#endif // NVCV_CUDA_DROP_CAST_HPP
