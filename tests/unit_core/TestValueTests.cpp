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

#include <common/ValueTests.hpp>

namespace t    = ::testing;
namespace test = nv::cv::test;

NVCV_TEST_SUITE_P(ValueTestsTests, test::ValueList{1, 2} * test::ValueList{'c', 'd'});

TEST_P(ValueTestsTests, test)
{
    // For now we're concerned if typed tests will compile.
    // TODO: How to test if the tests were correctly generated?
    int  p1 = GetParamValue<0>();
    char p2 = GetParamValue<1>();

    EXPECT_THAT(p1, t::AnyOf(1, 2));
    EXPECT_THAT(p2, t::AnyOf('c', 'd'));
}
