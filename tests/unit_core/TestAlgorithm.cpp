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

#include "Definitions.hpp"

#include <util/Algorithm.hpp>

namespace util = nv::cv::util;
namespace test = nv::cv::test;

TEST(AlgorithmMaxTest, works)
{
    EXPECT_EQ(2, util::Max(2));

    EXPECT_EQ(5, util::Max(2, 5));
    EXPECT_EQ(5, util::Max(5, 2));

    EXPECT_EQ(double('a'), util::Max(2, 3, 5.0, 'a'));
    EXPECT_EQ(7.6, util::Max(5, 7.6, 2, 3.0));
    EXPECT_EQ(5.0, util::Max(5, 2, 3.0));
    EXPECT_EQ(5.0, util::Max(5u, -2, 3.0));
}

template<int I>
struct Value
{
    constexpr static int value = I;
};

TEST(AlgorithmMaxTest, constexpr_works)
{
    Value<util::Max(3, 8)> v;
    EXPECT_EQ(8, v.value);
}
