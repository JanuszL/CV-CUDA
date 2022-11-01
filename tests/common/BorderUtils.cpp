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

void ReplicateBorderIndex(int2 &coord, int2 size)
{
    if (coord.x < 0)
    {
        coord.x = 0;
    }
    else
    {
        if (coord.x >= size.x)
        {
            coord.x = size.x - 1;
        }
    }

    if (coord.y < 0)
    {
        coord.y = 0;
    }
    else
    {
        if (coord.y >= size.y)
        {
            coord.y = size.y - 1;
        }
    }
}

void WrapBorderIndex(int2 &coord, int2 size)
{
    coord.x = coord.x % size.x;
    if (coord.x < 0)
    {
        coord.x += size.x;
    }

    coord.y = coord.y % size.y;
    if (coord.y < 0)
    {
        coord.y += size.y;
    }
}

void ReflectBorderIndex(int2 &coord, int2 size)
{
    const int2 last = {size.x - 1, size.y - 1};

    if (coord.x >= size.x)
    {
        coord.x = (last.x - std::abs(last.x - coord.x)) + 1;
    }
    else
    {
        coord.x = (last.x - std::abs(last.x - coord.x));
    }
    if (coord.x < 0)
    {
        coord.x = (std::abs(coord.x) - 1) % size.x;
    }
    else
    {
        coord.x = (std::abs(coord.x)) % size.x;
    }

    if (coord.y >= size.y)
    {
        coord.y = (last.y - std::abs(last.y - coord.y)) + 1;
    }
    else
    {
        coord.y = (last.y - std::abs(last.y - coord.y));
    }
    if (coord.y < 0)
    {
        coord.y = (std::abs(coord.y) - 1) % size.y;
    }
    else
    {
        coord.y = (std::abs(coord.y)) % size.y;
    }
}

void Reflect101BorderIndex(int2 &coord, int2 size)
{
    const int2 last = {size.x - 1, size.y - 1};

    coord.x = std::abs(last.x - std::abs(last.x - coord.x)) % size.x;

    coord.y = std::abs(last.y - std::abs(last.y - coord.y)) % size.y;
}

} // namespace nv::cv::test
