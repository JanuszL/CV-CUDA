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
 * @file Dims.hpp
 *
 * @brief Declaration of NVCV C++ Dims class and its operators.
 */

#ifndef NVCV_DIMS_HPP
#define NVCV_DIMS_HPP

#include <cassert>
#include <iostream>
#include <tuple>

namespace nv { namespace cv {

struct DimsNCHW
{
    int n, c, h, w;
};

inline bool operator==(const DimsNCHW &a, const DimsNCHW &b)
{
    return a.n == b.n && a.c == b.c && a.h == b.h && a.w == b.w;
}

inline bool operator!=(const DimsNCHW &a, const DimsNCHW &b)
{
    return !(a == b);
}

inline bool operator<(const DimsNCHW &a, const DimsNCHW &b)
{
    return std::tie(a.n, a.c, a.h, a.w) < std::tie(b.n, b.c, b.h, b.w);
}

inline std::ostream &operator<<(std::ostream &out, const DimsNCHW &dims)
{
    return out << "NCHW{" << dims.n << ',' << dims.c << ',' << dims.h << ',' << dims.w << '}';
}

}} // namespace nv::cv

#endif // NVCV_NCHW_HPP
