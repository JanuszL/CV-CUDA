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

#ifndef NVCV_UTIL_DIMS_HPP
#define NVCV_UTIL_DIMS_HPP
#include <tuple>

namespace nv::cv::util {

struct DimsNCHW
{
    int n, c, h, w;
};

inline bool operator==(const DimsNCHW &a, const DimsNCHW &b)
{
    return std::tie(a.n, a.c, a.h, a.w) == std::tie(b.n, b.c, b.h, b.w);
}

inline bool operator!=(const DimsNCHW &a, const DimsNCHW &b)
{
    return !(a == b);
}

} // namespace nv::cv::util

#endif // NVCV_UTIL_DIMS_HPP
