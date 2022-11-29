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

#ifndef NVCV_UTIL_ALGORITHM_HPP
#define NVCV_UTIL_ALGORITHM_HPP

namespace nv::cv::util {

template<class HEAD, class... TAIL>
constexpr auto Max(const HEAD &head, const TAIL &...tail)
{
    if constexpr (sizeof...(TAIL) == 0)
    {
        return head;
    }
    else
    {
        auto maxTail = Max(tail...);
        return head >= maxTail ? head : maxTail;
    }
}

} // namespace nv::cv::util

#endif // NVCV_UTIL_ALGORITHM_HPP
