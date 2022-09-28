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

#include <common/TypedTests.hpp>
#include <nvcv/detail/IndexSequence.hpp>

namespace test = nv::cv::test::type;
namespace nvcv = nv::cv;

// clang-format off
NVCV_TYPED_TEST_SUITE(IndexSequence,
    test::Types<
        test::Types<test::Value<0>, nvcv::detail::IndexSequence<>>,
        test::Types<test::Value<1>, nvcv::detail::IndexSequence<0>>,
        test::Types<test::Value<2>, nvcv::detail::IndexSequence<0,1>>,
        test::Types<test::Value<5>, nvcv::detail::IndexSequence<0,1,2,3,4>>
    >);

// clang-format on

TYPED_TEST(IndexSequence, make_index_sequence)
{
    constexpr int N = test::GetValue<TypeParam, 0>;
    using GOLD      = test::GetType<TypeParam, 1>;

    EXPECT_TRUE((std::is_same_v<GOLD, nvcv::detail::MakeIndexSequence<N>>));
}
