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
#include <util/Math.hpp>

namespace t    = ::testing;
namespace util = nv::cv::util;
namespace test = nv::cv::test;

class MathRoundUpTests
    : public t::TestWithParam<
          std::tuple<test::Param<"input", uint64_t>, test::Param<"next", int>, test::Param<"gold", uint64_t>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, MathRoundUpTests,
    test::ValueList<uint64_t, int, uint64_t>
    {
        {0, 1, 0},
        {1, 1, 1},
        {2, 1, 2},
        {3, 1, 3},

        {3, 5, 5},
        {4, 5, 5},
        {5, 5, 5},
        {6, 5, 10},
        {7, 5, 10},
        {9223372036854775802ull, 5, 9223372036854775805ull},

        {4, 6, 6},
        {5, 6, 6},
        {6, 6, 6},
        {7, 6, 12},
        {8, 6, 12},
        {9223372036854775802ull, 6, 9223372036854775806ull},

        {14, 16, 16},
        {15, 16, 16},
        {16, 16, 16},
        {17, 16, 32},
        {18, 16, 32},
        {9223372036854775502ull, 16, 9223372036854775504ull}
    });

// clang-format on

TEST_P(MathRoundUpTests, works)
{
    const uint64_t input = std::get<0>(GetParam());
    const int      next  = std::get<1>(GetParam());
    const uint64_t gold  = std::get<2>(GetParam());

    EXPECT_EQ(gold, util::RoundUp(input, next));
}

class MathIsPowerOfTwoTests
    : public t::TestWithParam<std::tuple<test::Param<"input", uint64_t>, test::Param<"gold", bool>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, MathIsPowerOfTwoTests,
    test::ValueList<uint64_t, bool>
    {
        {0, true},
        {1, true},
        {2, true},
        {4, true},

        {1ull << 30, true},
        {1ull << 31, true},
        {1ull << 32, true},
        {1ull << 62, true},
        {1ull << 63, true},

        {3, false},
        {6, false},
        {(1ull << 30) + 1, false},
        {(1ull << 31) + 1, false},
        {(1ull << 32) + 1, false},
        {(1ull << 62) + 1, false},
        {(1ull << 63) - 1, false}
    });

// clang-format on

TEST_P(MathIsPowerOfTwoTests, works)
{
    const uint64_t input = std::get<0>(GetParam());
    const bool     gold  = std::get<1>(GetParam());

    EXPECT_EQ(gold, util::IsPowerOfTwo(input));
}

class MathRoundUpNextPowerOfTwoTests
    : public t::TestWithParam<std::tuple<test::Param<"input", int64_t>, test::Param<"gold", int64_t>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, MathRoundUpNextPowerOfTwoTests,
    test::ValueList<int64_t, int64_t>
    {
        {0, 0},
        {1, 1},
        {2, 2},
        {3, 4},
        {4, 4},
        {37, 64},
        {125, 128},
        {128, 128},
        {129, 256},

        {251, 256},
        {256, 256},
        {257, 512},

        {32761, 32768},
        {32768, 32768},
        {32769, 65536},

        {(1ull << 31) - 1, 1ull << 31},
        {(1ull << 31)+1, 1ull << 32},
    });

// clang-format on

TEST_P(MathRoundUpNextPowerOfTwoTests, works)
{
    const int64_t input = std::get<0>(GetParam());
    const int64_t gold  = std::get<1>(GetParam());

    if (input < 128)
    {
        EXPECT_EQ(gold, util::RoundUpNextPowerOfTwo((int8_t)input));
    }
    else if (input < 256)
    {
        EXPECT_EQ(gold, util::RoundUpNextPowerOfTwo((uint8_t)input));
    }
    else if (input < 32768)
    {
        EXPECT_EQ(gold, util::RoundUpNextPowerOfTwo((int16_t)input));
    }
    else if (input < 65536)
    {
        EXPECT_EQ(gold, util::RoundUpNextPowerOfTwo((uint16_t)input));
    }
    else
    {
        EXPECT_EQ(gold, util::RoundUpNextPowerOfTwo(input));
    }
}
