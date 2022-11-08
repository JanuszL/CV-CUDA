/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvcv/Exception.hpp>

// TODO: once we have functions that generate errors, we should
// extend these tests to cover more scenarios

namespace t = ::testing;

TEST(ExceptionTest, exception_updates_internal_status)
{
    try
    {
        throw nv::cv::Exception(nv::cv::Status::ERROR_DEVICE, "test error");
    }
    catch (...)
    {
    }

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_ERROR_DEVICE, nvcvGetLastErrorMessage(msg, sizeof(msg)));
    EXPECT_STREQ("test error", msg);
}
