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

#include <nvcv/Version.h>

TEST(VersionTests, version_numeric)
{
    EXPECT_EQ(NVCV_VERSION, nvcvGetVersion());
}

TEST(VersionTests, get_version_components)
{
    EXPECT_EQ(NVCV_VERSION_MAJOR, nvcvGetVersion() / 1000000);
    EXPECT_EQ(NVCV_VERSION_MINOR, nvcvGetVersion() / 10000 % 100);
    EXPECT_EQ(NVCV_VERSION_PATCH, nvcvGetVersion() / 100 % 100);
    EXPECT_EQ(NVCV_VERSION_TWEAK, nvcvGetVersion() % 100);
}

// macro to stringify a macro-expanded expression
#define STR(a)        STR_HELPER(a)
#define STR_HELPER(a) #a

TEST(VersionTests, get_version_string)
{
    const char *ver = STR(NVCV_VERSION_MAJOR) "." STR(NVCV_VERSION_MINOR) "." STR(NVCV_VERSION_PATCH)
#if NVCV_VERSION_TWEAK
        "." STR(NVCV_VERSION_TWEAK)
#endif
            STR(NVCV_VERSION_SUFFIX);

    EXPECT_STREQ(ver, NVCV_VERSION_STRING);
}

TEST(VersionTests, make_version4_macro)
{
    EXPECT_EQ(1020304, NVCV_MAKE_VERSION4(1, 2, 3, 4));
}

TEST(VersionTests, make_version3_macro)
{
    EXPECT_EQ(1020300, NVCV_MAKE_VERSION3(1, 2, 3));
}

TEST(VersionTests, make_version2_macro)
{
    EXPECT_EQ(1020000, NVCV_MAKE_VERSION2(1, 2));
}

TEST(VersionTests, make_version1_macro)
{
    EXPECT_EQ(1000000, NVCV_MAKE_VERSION1(1));
}

TEST(VersionTests, make_version_macro)
{
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 2, 3, 4), NVCV_MAKE_VERSION(1, 2, 3, 4));
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 2, 3, 0), NVCV_MAKE_VERSION(1, 2, 3));
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 2, 0, 0), NVCV_MAKE_VERSION(1, 2));
    EXPECT_EQ(NVCV_MAKE_VERSION4(1, 0, 0, 0), NVCV_MAKE_VERSION(1));
}

TEST(VersionTests, api_version_macro)
{
    EXPECT_EQ(NVCV_MAKE_VERSION(NVCV_VERSION_MAJOR, NVCV_VERSION_MINOR), NVCV_VERSION_API);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"

#undef NVCV_VERSION_API
#define NVCV_VERSION_API NVCV_MAKE_VERSION(1, 0)

#pragma GCC diagnostic pop

TEST(VersionTests, api_version_macro_redefinition)
{
    EXPECT_EQ(NVCV_MAKE_VERSION(1, 0), NVCV_VERSION_API);
}

TEST(VersionTests, api_version_at_least)
{
    EXPECT_TRUE(NVCV_VERSION_API_AT_LEAST(1, 0));
    EXPECT_TRUE(NVCV_VERSION_API_AT_LEAST(0, 99));

    EXPECT_FALSE(NVCV_VERSION_API_AT_LEAST(1, 2));
    EXPECT_FALSE(NVCV_VERSION_API_AT_LEAST(2, 0));
}

TEST(VersionTests, api_version_at_most)
{
    EXPECT_TRUE(NVCV_VERSION_API_AT_MOST(1, 0));
    EXPECT_FALSE(NVCV_VERSION_API_AT_MOST(0, 99));
    EXPECT_TRUE(NVCV_VERSION_API_AT_MOST(1, 1));
    EXPECT_TRUE(NVCV_VERSION_API_AT_MOST(2, 0));
}

TEST(VersionTests, api_version_in_range)
{
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(1, 0, 1, 0));
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(0, 99, 1, 0));
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(0, 99, 1, 1));
    EXPECT_TRUE(NVCV_VERSION_API_IN_RANGE(0, 99, 2, 0));

    EXPECT_FALSE(NVCV_VERSION_API_IN_RANGE(0, 98, 0, 99));
    EXPECT_FALSE(NVCV_VERSION_API_IN_RANGE(1, 1, 1, 2));
}

TEST(VersionTests, api_version_is)
{
    EXPECT_TRUE(NVCV_VERSION_API_IS(1, 0));
    EXPECT_FALSE(NVCV_VERSION_API_IS(1, 1));
    EXPECT_FALSE(NVCV_VERSION_API_IS(0, 99));
}
