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
#include <util/Metaprogramming.hpp>

namespace ttest = nv::cv::test::type;
namespace util  = nv::cv::util;

// clang-format off
NVCV_TYPED_TEST_SUITE(MetaprogrammingTypeIdentityTest,
                      ttest::Types<
                        ttest::Types<int, int>,
                        ttest::Types<char & , char &>,
                        ttest::Types<const short * , const short *>,
                        ttest::Types<volatile long *, volatile long *>,
                        ttest::Types<volatile long *, volatile long *>,
                        ttest::Types<const int*[3], const int*[3]>>);

// clang-format on

TYPED_TEST(MetaprogrammingTypeIdentityTest, works)
{
    using IN   = ttest::GetType<TypeParam, 0>;
    using GOLD = ttest::GetType<TypeParam, 1>;

    EXPECT_TRUE((std::is_same_v<util::TypeIdentity<IN>, GOLD>));
}
