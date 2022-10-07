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

#include <nvcv/detail/Optional.hpp>

namespace d = nv::cv::detail;

TEST(Optional, default_no_value)
{
    d::Optional<int> opt;

    EXPECT_FALSE(opt.hasValue());
    EXPECT_FALSE(opt && true);

    EXPECT_THROW(*opt, std::runtime_error);
}

TEST(Optional, ctor_with_value)
{
    d::Optional<int> opt(5);

    EXPECT_TRUE(opt.hasValue());
    EXPECT_TRUE(opt && true);

    EXPECT_EQ(5, *opt);
}

// TODO need way more tests.
// We're not writing them now because if we can upgrade public API to c++17,
// we won't need our Optional.
