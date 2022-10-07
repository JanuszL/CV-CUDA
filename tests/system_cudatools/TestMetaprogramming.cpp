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
