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

#include <common/MixTypedTests.hpp>

namespace t    = ::testing;
namespace test = nv::cv::test;

// clang-format off
NVCV_MIXTYPED_TEST_SUITE(TypedMixedTest,
    test::type::Combine<test::Types<test::type::Value<0>, test::type::Value<2>, test::type::Value<3>>,
                                    test::Types<int,float,char>,
                                    test::Types<test::type::Value<'a'>, test::type::Value<'b'>>
    >);

// clang-format on

NVCV_MIXTYPED_TEST(TypedMixedTest, test)
{
    const int A  = GetValue<0>();
    using T      = GetType<1>;
    const char B = GetValue<2>();

    EXPECT_THAT(A, t::AnyOf(0, 2, 3));
    EXPECT_TRUE((std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, char>));
    EXPECT_THAT(B, t::AnyOf('a', 'b'));
}
