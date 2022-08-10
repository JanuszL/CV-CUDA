/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gtest/gtest.h>
#include <cvcuda/Foo.hpp>

namespace cuda = nv::cuda;

TEST(FooTest, works)
{
    EXPECT_TRUE(cuda::Foo(42));
    EXPECT_FALSE(cuda::Foo(41));
    EXPECT_FALSE(cuda::Foo(43));
}
