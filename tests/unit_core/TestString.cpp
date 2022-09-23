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
#include <util/Assert.h>
#include <util/String.hpp>

namespace util = nv::cv::util;
namespace test = nv::cv::test;
namespace t    = ::testing;

NVCV_TEST_SUITE_P(ReplaceAllInlineTests,
                  // input, bufSize, what, replace, gold
                  test::ValueList<const char *, int, const char *, const char *, const char *>{
                      {      "test", 16, "ROD",    "lima",         "test"}, // pattern not found

                      {   "RODtest", 16, "ROD",        "",         "test"}, // remove pattern once begin
                      {   "teRODst", 16, "ROD",        "",         "test"}, // remove pattern once middle
                      {   "testROD", 16, "ROD",        "",         "test"}, // remove pattern once end
                      {"tRODeRODst", 16, "ROD",        "",         "test"}, // remove pattern twice

                      {   "RODtest", 16, "ROD",      "AB",       "ABtest"}, // replace pattern with smaller once begin
                      {   "teRODst", 16, "ROD",      "AB",       "teABst"}, // replace pattern with smaller once middle
                      {   "testROD", 16, "ROD",      "AB",       "testAB"}, // replace pattern with smaller once end
                      {"tRODeRODst", 16, "ROD",      "AB",     "tABeABst"}, // replace pattern with smaller twice

                      {   "RODtest", 16, "ROD",    "ABCD",     "ABCDtest"}, // replace pattern with larger once begin
                      {   "teRODst", 16, "ROD",    "ABCD",     "teABCDst"}, // replace pattern with larger once middle
                      {   "testROD", 16, "ROD",    "ABCD",     "testABCD"}, // replace pattern with larger once end
                      {"tRODeRODst", 16, "ROD",    "ABCD", "tABCDeABCDst"}, // replace pattern with larger twice

                      {   "tRODest",  7, "ROD",    "ABCD",      "tABCDes"}, // buffer size too small for replacement

                      {   "tRODest", 32, "ROD", "RODOLFO",  "tRODOLFOest"}, // 'replacement' contains 'what'
});

TEST_P(ReplaceAllInlineTests, test)
{
    const char *input   = GetParamValue<0>();
    const int   bufSize = GetParamValue<1>();
    const char *what    = GetParamValue<2>();
    const char *replace = GetParamValue<3>();
    const char *gold    = GetParamValue<4>();

    char buffer[256];
    // +1 for sentinel
    NVCV_ASSERT(sizeof(buffer) + 1 >= strlen(input));
    NVCV_ASSERT(sizeof(buffer) + 1 >= strlen(gold));

    strncpy(buffer, input, sizeof(buffer));
    char *sentinel = buffer + std::max(strlen(input), strlen(gold)) + 1;
    *sentinel      = '\xFF';

    ASSERT_NO_THROW(util::ReplaceAllInline(buffer, bufSize, what, replace));

    EXPECT_STREQ(gold, buffer);
    EXPECT_EQ('\xFF', *sentinel) << "buffer overrun";
}
