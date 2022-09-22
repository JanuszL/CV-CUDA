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
#include <private/core/Version.hpp>

namespace t    = ::testing;
namespace priv = nv::cv::priv;
namespace test = nv::cv::test;

#define MAKE_VERSION(major, minor, patch, tweak) ((major)*1000000 + (minor)*10000 + (patch)*100 + (tweak))

class VersionTests
    : public t::TestWithParam<std::tuple<test::Param<"code", uint32_t>,             // 0
                                         test::Param<"major", int>,                 // 1
                                         test::Param<"minor", int>,                 // 2
                                         test::Param<"patch", int>,                 // 3
                                         test::Param<"tweak", int>,                 // 4
                                         test::Param<"ctor_result", NVCVStatus>,    // 5
                                         test::Param<"values_result", NVCVStatus>>> // 6
{
public:
    VersionTests()
        : m_paramCode(std::get<0>(GetParam()))
        , m_paramMajor(std::get<1>(GetParam()))
        , m_paramMinor(std::get<2>(GetParam()))
        , m_paramPatch(std::get<3>(GetParam()))
        , m_paramTweak(std::get<4>(GetParam()))
        , m_goldCtorResult(std::get<5>(GetParam()))
        , m_goldValuesResult(std::get<6>(GetParam()))
    {
    }

protected:
    uint32_t   m_paramCode;
    int        m_paramMajor;
    int        m_paramMinor;
    int        m_paramPatch;
    int        m_paramTweak;
    NVCVStatus m_goldCtorResult;
    NVCVStatus m_goldValuesResult;
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(Positive, VersionTests,
                              test::ValueList<uint32_t, int, int, int, int>
                              {
                                  { MAKE_VERSION(0,  0,  0,  0),  0,  0,  0,  0},
                                  { MAKE_VERSION(1,  0,  0,  0),  1,  0,  0,  0},
                                  { MAKE_VERSION(0,  1,  0,  0),  0,  1,  0,  0},
                                  { MAKE_VERSION(0,  0,  1,  0),  0,  0,  1,  0},
                                  { MAKE_VERSION(0,  0,  0,  1),  0,  0,  0,  1},
                                  { MAKE_VERSION(1,  2,  3,  4),  1,  2,  3,  4},
                                  { MAKE_VERSION(3,  2,  1,  0),  3,  2,  1,  0},
                                  {MAKE_VERSION(99,  0,  0,  0), 99,  0,  0,  0},
                                  { MAKE_VERSION(0, 99,  0,  0),  0, 99,  0,  0},
                                  { MAKE_VERSION(0,  0, 99,  0),  0,  0, 99,  0},
                                  { MAKE_VERSION(0,  0,  0, 99),  0,  0,  0, 99},
                                  {MAKE_VERSION(99, 99, 99, 99), 99, 99, 99, 99},
                                  {MAKE_VERSION(100, 99, 99, 99), 100, 99, 99, 99},
                                  {MAKE_VERSION(500, 99, 99, 99), 500, 99, 99, 99},
                                } * NVCV_SUCCESS * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_ctor_fail, VersionTests,
                              test::ValueList<uint32_t, int, int, int, int>
                              {
                                  { MAKE_VERSION(-1,   0,   0,   0),  -1,   0,   0,   0},
                                  {  MAKE_VERSION(0,  -1,   0,   0),   0,  -1,   0,   0},
                                  {  MAKE_VERSION(0,   0,  -1,   0),   0,   0,  -1,   0},
                                  {  MAKE_VERSION(0,   0,   0,  -1),   0,   0,   0,  -1},
                                  { MAKE_VERSION(-4,  -2,  -9,   4),  -4,  -2,  -9,   4},
                                  {MAKE_VERSION(105, 102, 120, 150), 105, 102, 120, 150},
                              } * NVCV_ERROR_INVALID_ARGUMENT * NVCV_ERROR_INVALID_ARGUMENT);

NVCV_INSTANTIATE_TEST_SUITE_P(Negative_values_fail, VersionTests,
                              test::ValueList<uint32_t, int, int, int, int>
                              {
                                  {MAKE_VERSION(0, 100,   0,   0), 0, 100,   0,   0},
                                  {MAKE_VERSION(0,   0, 100,   0), 0,   0, 100,   0},
                                  {MAKE_VERSION(0,   0,   0, 100), 0,   0,   0, 100},
                              } * NVCV_SUCCESS * NVCV_ERROR_INVALID_ARGUMENT);

// clang-format on

TEST_P(VersionTests, code_to_version)
{
    std::unique_ptr<priv::Version> ver;

    // Code is always valid because any overflow will just make at most the major version
    // larger than expected. We can't check for that, it's still a valid version.
    NVCV_ASSERT_STATUS(NVCV_SUCCESS, ver = std::make_unique<priv::Version>(m_paramCode));

    if (m_goldValuesResult == NVCV_SUCCESS)
    {
        ASSERT_NE(nullptr, ver);

        EXPECT_EQ(ver->major(), m_paramMajor);
        EXPECT_EQ(ver->minor(), m_paramMinor);
        EXPECT_EQ(ver->patch(), m_paramPatch);
        EXPECT_EQ(ver->tweak(), m_paramTweak);
    }
    else if (ver && m_goldCtorResult == NVCV_SUCCESS)
    {
        EXPECT_TRUE(ver->major() != m_paramMajor || ver->minor() != m_paramMinor || ver->patch() != m_paramPatch
                    || ver->tweak() != m_paramTweak);
    }
}

TEST_P(VersionTests, version_to_code)
{
    std::unique_ptr<priv::Version> ver;
    NVCV_ASSERT_STATUS(m_goldValuesResult,
                       ver = std::make_unique<priv::Version>(m_paramMajor, m_paramMinor, m_paramPatch, m_paramTweak));

    if (ver)
    {
        EXPECT_EQ(m_paramCode, ver->code());
    }
}

class VersionStringTests
    : public t::TestWithParam<std::tuple<test::Param<"major", int>,            // 0
                                         test::Param<"minor", int>,            // 1
                                         test::Param<"patch", int>,            // 2
                                         test::Param<"tweak", int>,            // 3
                                         test::Param<"result", const char *>>> // 4
{
public:
    VersionStringTests()
        : m_paramMajor(std::get<0>(GetParam()))
        , m_paramMinor(std::get<1>(GetParam()))
        , m_paramPatch(std::get<2>(GetParam()))
        , m_paramTweak(std::get<3>(GetParam()))
        , m_goldResult(std::get<4>(GetParam()))
    {
    }

protected:
    int         m_paramMajor;
    int         m_paramMinor;
    int         m_paramPatch;
    int         m_paramTweak;
    const char *m_goldResult;
};

NVCV_INSTANTIATE_TEST_SUITE_P(Positive, VersionStringTests,
                              test::ValueList<int, int, int, int, const char *>{
                                  { 0,  0,  0,  0,       "v0.0.0"},
                                  { 1,  2,  3,  4,     "v1.2.3.4"},
                                  { 1,  2,  3,  0,       "v1.2.3"},
                                  {99, 99, 99, 99, "v99.99.99.99"},
                                  {99, 99, 99,  0,    "v99.99.99"}
});

TEST_P(VersionStringTests, test)
{
    priv::Version ver(m_paramMajor, m_paramMinor, m_paramPatch, m_paramTweak);

    std::ostringstream ss;
    ss << ver;

    EXPECT_STREQ(m_goldResult, ss.str().c_str());
}

class VersionComparisonTests
    : public t::TestWithParam<std::tuple<test::Param<"lhs", priv::Version, priv::Version{0, 0, 0, 0}>, // 0
                                         test::Param<"rhs", priv::Version, priv::Version{0, 0, 0, 0}>, // 1
                                         test::Param<"result", int>>>                                  // 2
{
public:
    VersionComparisonTests()
        : m_paramLHS(std::get<0>(GetParam()))
        , m_paramRHS(std::get<1>(GetParam()))
        , m_goldResult(std::get<2>(GetParam()))
    {
    }

protected:
    priv::Version m_paramLHS, m_paramRHS;
    int           m_goldResult;
};

NVCV_INSTANTIATE_TEST_SUITE_P(Positive, VersionComparisonTests,
                              test::ValueList<priv::Version, priv::Version, int>{
                                  {priv::Version{1, 0, 0}, priv::Version{0, 1, 0}, 1},
                                  {priv::Version{1, 2, 3}, priv::Version{1, 2, 3}, 0},
                                  {priv::Version{1, 0, 0}, priv::Version{0, 0, 1}, 1},
});

TEST_P(VersionComparisonTests, lower_than)
{
    EXPECT_EQ(m_goldResult < 0, m_paramLHS < m_paramRHS);
}

TEST_P(VersionComparisonTests, lower_equal_than)
{
    EXPECT_EQ(m_goldResult <= 0, m_paramLHS <= m_paramRHS);
}

TEST_P(VersionComparisonTests, equal_than)
{
    EXPECT_EQ(m_goldResult == 0, m_paramLHS == m_paramRHS);
}

TEST_P(VersionComparisonTests, not_equal_than)
{
    EXPECT_EQ(m_goldResult != 0, m_paramLHS != m_paramRHS);
}

TEST_P(VersionComparisonTests, greater_equal_than)
{
    EXPECT_EQ(m_goldResult >= 0, m_paramLHS >= m_paramRHS);
}

TEST_P(VersionComparisonTests, greater_than)
{
    EXPECT_EQ(m_goldResult > 0, m_paramLHS > m_paramRHS);
}
