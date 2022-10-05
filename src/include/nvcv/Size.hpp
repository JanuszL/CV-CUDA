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
 * @file Size.hpp
 *
 * @brief Declaration of NVCV C++ Size class and its operators.
 */

#ifndef NVCV_SIZE_HPP
#define NVCV_SIZE_HPP

#include <cassert>
#include <iostream>

namespace nv { namespace cv {

/**
 * @defgroup NVCV_CPP_CORE_SIZE Size Operator
 * @{
*/

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

inline bool operator<(const Size2D &a, const Size2D &b)
{
    int64_t areaA = static_cast<int64_t>(a.w) * a.h;
    int64_t areaB = static_cast<int64_t>(b.w) * b.h;

    if (areaA == areaB)
    {
        if (a.w == b.w)
        {
            assert(a.h == b.h);
            return false;
        }
        else
        {
            return a.w < b.w;
        }
    }
    else
    {
        return areaA < areaB;
    }
}

inline std::ostream &operator<<(std::ostream &out, const Size2D &size)
{
    return out << size.w << "x" << size.h;
}

/**@}*/

}} // namespace nv::cv

#endif // NVCV_SIZE_HPP
