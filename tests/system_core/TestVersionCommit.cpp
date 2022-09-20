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

#include <nvcv/detail/VersionUtils.h> // for NVCV_COMMIT

TEST(VersionTest, commit_hash_macro_exists)
{
#ifdef NVCV_COMMIT
    // Just some random, valid commit hash
    EXPECT_EQ(strlen("5335ae6bb8161d8e7f05896288f289cd84517e47"), strlen(NVCV_COMMIT));
#else
    FAIL() << "NVCV_COMMIT not defined";
#endif
}
