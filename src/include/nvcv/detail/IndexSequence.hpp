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

#ifndef NVCV_DETAIL_INDEXSEQUENCE_HPP
#define NVCV_DETAIL_INDEXSEQUENCE_HPP

#include <cstddef>
#include <type_traits>

namespace nv { namespace cv { namespace detail {

template<std::size_t... II>
struct IndexSequence
{
};

template<std::size_t N, std::size_t... II>
struct MakeIndexSequenceImpl
{
    using type = typename MakeIndexSequenceImpl<N - 1, N, II...>::type;
};

template<std::size_t... II>
struct MakeIndexSequenceImpl<0, II...>
{
    using type = IndexSequence<(II - 1)...>;
};

template<std::size_t N>
using MakeIndexSequence = typename MakeIndexSequenceImpl<N>::type;

}}} // namespace nv::cv::detail

#endif // NVCV_DETAIL_INDEXSEQUENCE_HPP
