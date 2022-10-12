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
#include <nvcv/Dims.hpp>

namespace gt   = ::testing;
namespace test = nv::cv::test;
namespace nvcv = nv::cv;

// DimsNCHW Equality --------------------------------------------
class DimsNCHWEqualityTests : public gt::TestWithParam<std::tuple<nvcv::DimsNCHW, nvcv::DimsNCHW, bool>>
{
};

// We can't use ValueLists because its test instantiation uses Dims's operator==,
// we shouldn't be using something we're testing.

// clang-format off
INSTANTIATE_TEST_SUITE_P(Positive, DimsNCHWEqualityTests,
                         gt::Values(std::make_tuple(nvcv::DimsNCHW{1,5,3,6}, nvcv::DimsNCHW{1,5,3,6}, true),
                                    std::make_tuple(nvcv::DimsNCHW{2,5,-1,-55}, nvcv::DimsNCHW{2,5,-1,-55}, true),
                                    std::make_tuple(nvcv::DimsNCHW{0,4,4232,-1853}, nvcv::DimsNCHW{0,4,4232, -1853}, true)));

INSTANTIATE_TEST_SUITE_P(Negative, DimsNCHWEqualityTests,
                         gt::Values(std::make_tuple(nvcv::DimsNCHW{2,5,3,6}, nvcv::DimsNCHW{1,5,3,6}, false),
                                    std::make_tuple(nvcv::DimsNCHW{1,4,3,6}, nvcv::DimsNCHW{1,5,3,6}, false),
                                    std::make_tuple(nvcv::DimsNCHW{1,5,2,6}, nvcv::DimsNCHW{1,5,3,6}, false),
                                    std::make_tuple(nvcv::DimsNCHW{1,5,3,9}, nvcv::DimsNCHW{1,5,3,6}, false)));

// clang-format on

TEST_P(DimsNCHWEqualityTests, are_equal)
{
    nvcv::DimsNCHW a     = std::get<0>(GetParam());
    nvcv::DimsNCHW b     = std::get<1>(GetParam());
    bool           equal = std::get<2>(GetParam());

    EXPECT_EQ(equal, a == b);
    EXPECT_EQ(!equal, a != b);
}

// DimsNCHW ordering --------------------------------------------
class DimsNCHWLessThanTests : public gt::TestWithParam<std::tuple<nvcv::DimsNCHW, nvcv::DimsNCHW, bool>>
{
};

// We can't use ValueLists because its test instantiation uses DimsNCHW's operator<,
// we shouldn't be using something we're testing.

// clang-format off
INSTANTIATE_TEST_SUITE_P(Positive, DimsNCHWLessThanTests,
                         gt::Values(std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{3,5,3,6}, true),
                                    std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{2,5,2,6}, true),
                                    std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{2,4,3,7}, true)));

INSTANTIATE_TEST_SUITE_P(Negative, DimsNCHWLessThanTests,
                         gt::Values(std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{2,4,3,6}, false),
                                    std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{1,4,3,6}, false),
                                    std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{2,3,3,6}, false),
                                    std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{2,4,2,6}, false),
                                    std::make_tuple(nvcv::DimsNCHW{2,4,3,6}, nvcv::DimsNCHW{2,4,3,5}, false)));

// clang-format on

TEST_P(DimsNCHWLessThanTests, is_less_than)
{
    nvcv::DimsNCHW a        = std::get<0>(GetParam());
    nvcv::DimsNCHW b        = std::get<1>(GetParam());
    bool           lessThan = std::get<2>(GetParam());

    EXPECT_EQ(lessThan, a < b);
}

// DimsNCHW print --------------------------------------------

static test::ValueList<nvcv::DimsNCHW, const char *> g_DimsNCHWNames = {
    {      nvcv::DimsNCHW{1, 5, 7, 0},       "NCHW{1,5,7,0}"},
    {   nvcv::DimsNCHW{2, 5, 132, 19},    "NCHW{2,5,132,19}"},
    {nvcv::DimsNCHW{-4, -5, -42, -28}, "NCHW{-4,-5,-42,-28}"}
};

NVCV_TEST_SUITE_P(DimsNCHWPrintTests, g_DimsNCHWNames);

TEST_P(DimsNCHWPrintTests, print_size)
{
    nvcv::DimsNCHW s    = GetParamValue<0>();
    const char    *gold = GetParamValue<1>();

    std::ostringstream ss;
    ss << s;

    ASSERT_STREQ(gold, ss.str().c_str());
}
