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

#include <common/TypedTests.hpp>                // for NVCV_TYPED_TEST_SUITE_F, etc.
#include <nvcv/cuda/detail/Metaprogramming.hpp> // the object of this test

namespace t      = ::testing;
namespace test   = nv::cv::test;
namespace detail = nv::cv::cuda::detail;

// ------------------------- Testing CopyConstness_t ---------------------------

template<typename T>
class CopyConstnessTest : public t::Test
{
public:
    using SourceType = test::type::GetType<T, 0>;
    using TargetType = test::type::GetType<T, 1>;
};

NVCV_TYPED_TEST_SUITE_F(CopyConstnessTest,
                        test::type::Zip<t::Types<const float, const unsigned long long>, t::Types<int, double>>);

TYPED_TEST(CopyConstnessTest, CorrectConstType)
{
    using ConstType = detail::CopyConstness_t<typename TestFixture::SourceType, typename TestFixture::TargetType>;

    EXPECT_TRUE(std::is_const_v<ConstType>);
}

TYPED_TEST(CopyConstnessTest, CorrectTargetType)
{
    using ConstType = detail::CopyConstness_t<typename TestFixture::SourceType, typename TestFixture::TargetType>;

    EXPECT_TRUE((std::is_same_v<typename std::remove_const_t<ConstType>, typename TestFixture::TargetType>));
}

// ------------------------- Testing HasTypeTraits -----------------------------

template<typename T>
class HasTypeTraitsTest : public t::Test
{
public:
    using Type = T;
};

template<typename T>
class HasTypeTraitsUnsupportedTest : public HasTypeTraitsTest<T>
{
};

typedef struct _float5
{
    float a, b, c, d, e;
} float5;

using UnsupportedBaseTypes = t::Types<void, long double, float5>;

TYPED_TEST_SUITE(HasTypeTraitsUnsupportedTest, UnsupportedBaseTypes);

TYPED_TEST(HasTypeTraitsUnsupportedTest, IsFalse)
{
    EXPECT_FALSE(detail::HasTypeTraits<typename TestFixture::Type>);
}

TEST(HasTypeTraitsWithTwoUnsupportedTypesTest, IsFalse)
{
    EXPECT_FALSE((detail::HasTypeTraits<long double, float5>));
}

template<typename T>
class HasTypeTraitsSupportedTest : public HasTypeTraitsTest<T>
{
};

using SupportedBaseTypes = t::Types<int, float3>;

TYPED_TEST_SUITE(HasTypeTraitsSupportedTest, SupportedBaseTypes);

TYPED_TEST(HasTypeTraitsSupportedTest, IsTrue)
{
    EXPECT_TRUE(detail::HasTypeTraits<typename TestFixture::Type>);
}

TEST(HasTypeTraitsWithTwoSupportedTypesTest, IsTrue)
{
    EXPECT_TRUE((detail::HasTypeTraits<unsigned int, double4>));
}

// -------------------------- Testing IsCompound -------------------------------

TEST(IsCompoundTest, IsFalse)
{
    EXPECT_FALSE((detail::IsCompound<unsigned char>));
    EXPECT_FALSE((detail::IsCompound<char>));
    EXPECT_FALSE((detail::IsCompound<unsigned short>));
    EXPECT_FALSE((detail::IsCompound<int>));
}

TEST(IsCompoundTest, IsTrue)
{
    EXPECT_TRUE((detail::IsCompound<uchar1>));
    EXPECT_TRUE((detail::IsCompound<short2>));
    EXPECT_TRUE((detail::IsCompound<uint3>));
    EXPECT_TRUE((detail::IsCompound<float4>));
}
