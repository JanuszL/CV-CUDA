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

#include <common/TypeList.hpp>

namespace t    = ::testing;
namespace test = nv::cv::test;

struct Foo
{
};

struct Bar
{
};

NVCV_TYPED_TEST_SUITE(TypedTest, test::type::Combine<test::Types<Foo, Bar>, test::Values<1, 2, 3>>);

TYPED_TEST(TypedTest, test)
{
    // For now we're concerned if typed tests will compile.
    // TODO: How to test if the tests were correctly generated?
}
