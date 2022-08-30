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

#ifndef NVCV_UTIL_SIZE_HPP
#define NVCV_UTIL_SIZE_HPP

namespace nv::cv::util {

struct Size2D
{
    int w, h;
};

inline bool operator==(const Size2D &a, const Size2D &b)
{
    return a.w == b.w && a.h == b.h;
}

inline bool operator!=(const Size2D &a, const Size2D &b)
{
    return !(a == b);
}

} // namespace nv::cv::util

#endif // NVCV_UTIL_SIZE_HPP
