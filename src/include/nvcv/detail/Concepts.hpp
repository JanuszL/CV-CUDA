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

#ifndef NVCV_DETAIL_CONCEPTS_HPP
#define NVCV_DETAIL_CONCEPTS_HPP

#include <iterator>
#include <type_traits>

namespace nv { namespace cv { namespace detail {

template<class IT>
using IsRandomAccessIterator = typename std::enable_if<
    std::is_same<typename std::iterator_traits<IT>::iterator_category, std::random_access_iterator_tag>::value>::type;

}}} // namespace nv::cv::detail

#endif // NVCV_DETAIL_CONCEPTS_HPP
