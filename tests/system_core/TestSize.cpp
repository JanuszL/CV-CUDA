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
#include <nvcv/Size.hpp>

namespace gt   = ::testing;
namespace test = nv::cv::test;
namespace nvcv = nv::cv;

// Size2D Equality --------------------------------------------
class Size2DEqualityTests : public gt::TestWithParam<std::tuple<nvcv::Size2D, nvcv::Size2D, bool>>
{
};

// We can't use ValueLists because its test instantiation uses Size's operator==,
// we shouldn't be using something we're testing.

// clang-format off
INSTANTIATE_TEST_SUITE_P(Positive, Size2DEqualityTests,
                         gt::Values(std::make_tuple(nvcv::Size2D{1,5}, nvcv::Size2D{1,5}, true),
                                    std::make_tuple(nvcv::Size2D{2,5}, nvcv::Size2D{2,5}, true),
                                    std::make_tuple(nvcv::Size2D{0,4}, nvcv::Size2D{0,4}, true)));

INSTANTIATE_TEST_SUITE_P(Negative, Size2DEqualityTests,
                         gt::Values(std::make_tuple(nvcv::Size2D{1,5}, nvcv::Size2D{2,5}, false),
                                    std::make_tuple(nvcv::Size2D{1,5}, nvcv::Size2D{1,4}, false),
                                    std::make_tuple(nvcv::Size2D{2,5}, nvcv::Size2D{1,5}, false),
                                    std::make_tuple(nvcv::Size2D{2,6}, nvcv::Size2D{1,5}, false)));

// clang-format on

TEST_P(Size2DEqualityTests, are_equal)
{
    nvcv::Size2D a     = std::get<0>(GetParam());
    nvcv::Size2D b     = std::get<1>(GetParam());
    bool         equal = std::get<2>(GetParam());

    EXPECT_EQ(equal, a == b);
    EXPECT_EQ(!equal, a != b);
}

// Size2D ordering --------------------------------------------
class Size2DLessThanTests : public gt::TestWithParam<std::tuple<nvcv::Size2D, nvcv::Size2D, bool>>
{
};

// We can't use ValueLists because its test instantiation uses Size2D's operator<,
// we shouldn't be using something we're testing.

// clang-format off
INSTANTIATE_TEST_SUITE_P(Positive, Size2DLessThanTests,
                         gt::Values(std::make_tuple(nvcv::Size2D{2,4}, nvcv::Size2D{2,6}, true),
                                    std::make_tuple(nvcv::Size2D{3,3}, nvcv::Size2D{2,6}, true),
                                    std::make_tuple(nvcv::Size2D{3,4}, nvcv::Size2D{3,6}, true),
                                    std::make_tuple(nvcv::Size2D{3,4}, nvcv::Size2D{2,7}, true)));

INSTANTIATE_TEST_SUITE_P(Negative, Size2DLessThanTests,
                         gt::Values(std::make_tuple(nvcv::Size2D{3,4}, nvcv::Size2D{2,6}, false), // same area
                                    std::make_tuple(nvcv::Size2D{4,4}, nvcv::Size2D{2,6}, false),
                                    std::make_tuple(nvcv::Size2D{3,5}, nvcv::Size2D{2,6}, false),
                                    std::make_tuple(nvcv::Size2D{3,4}, nvcv::Size2D{1,6}, false),
                                    std::make_tuple(nvcv::Size2D{3,4}, nvcv::Size2D{2,5}, false)));

// clang-format on

TEST_P(Size2DLessThanTests, is_less_than)
{
    nvcv::Size2D a        = std::get<0>(GetParam());
    nvcv::Size2D b        = std::get<1>(GetParam());
    bool         lessThan = std::get<2>(GetParam());

    EXPECT_EQ(lessThan, a < b);
}

// Size2D print --------------------------------------------

static test::ValueList<nvcv::Size2D, const char *> g_Size2DNames = {
    { nvcv::Size2D{1, 5},  "1x5"},
    { nvcv::Size2D{2, 5},  "2x5"},
    { nvcv::Size2D{2, 8},  "2x8"},
    {nvcv::Size2D{-4, 5}, "-4x5"},
    {nvcv::Size2D{4, -5}, "4x-5"}
};

NVCV_TEST_SUITE_P(Size2DPrintTests, g_Size2DNames);

TEST_P(Size2DPrintTests, print_size)
{
    nvcv::Size2D s    = GetParamValue<0>();
    const char  *gold = GetParamValue<1>();

    std::ostringstream ss;
    ss << s;

    ASSERT_STREQ(gold, ss.str().c_str());
}
