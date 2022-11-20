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

#include "BorderUtils.hpp"

#include <cmath>

namespace nv::cv::test {

inline void ReplicateBorderIndex(int &coord, int size)
{
    if (coord < 0)
    {
        coord = 0;
    }
    else
    {
        if (coord >= size)
        {
            coord = size - 1;
        }
    }
}

inline void WrapBorderIndex(int &coord, int size)
{
    coord = coord % size;
    if (coord < 0)
    {
        coord += size;
    }
}

inline void ReflectBorderIndex(int &coord, int size)
{
    // Reflect 1001: starting at size, we slope downards, the value at size - 1 is repeated
    coord = coord % (size * 2);
    if (coord < 0)
    {
        coord += size * 2;
    }
    if (coord >= size)
    {
        coord = size - 1 - (coord - size);
    }
}

inline void Reflect101BorderIndex(int &coord, int size)
{
    coord = coord % (2 * size - 2);
    if (coord < 0)
    {
        coord += 2 * size - 2;
    }
    coord = size - 1 - abs(size - 1 - coord);
}

void ReplicateBorderIndex(int2 &coord, int2 size)
{
    ReplicateBorderIndex(coord.x, size.x);
    ReplicateBorderIndex(coord.y, size.y);
}

void WrapBorderIndex(int2 &coord, int2 size)
{
    WrapBorderIndex(coord.x, size.x);
    WrapBorderIndex(coord.y, size.y);
}

void ReflectBorderIndex(int2 &coord, int2 size)
{
    ReflectBorderIndex(coord.x, size.x);
    ReflectBorderIndex(coord.y, size.y);
}

void Reflect101BorderIndex(int2 &coord, int2 size)
{
    Reflect101BorderIndex(coord.x, size.x);
    Reflect101BorderIndex(coord.y, size.y);
}

} // namespace nv::cv::test
