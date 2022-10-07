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

#include <nvcv/Status.h>

// TODO: once we have functions that generate errors, we should
// extend these tests to cover more scenarios

namespace t = ::testing;

class StatusNameTest : public t::TestWithParam<std::tuple<NVCVStatus, const char *>>
{
};

#define MAKE_STATUS_NAME(X) std::make_tuple(X, #X)

// clang-format off
INSTANTIATE_TEST_SUITE_P(AllStatuses, StatusNameTest,
                         t::Values(MAKE_STATUS_NAME(NVCV_SUCCESS),
                                    MAKE_STATUS_NAME(NVCV_ERROR_NOT_IMPLEMENTED),
                                    MAKE_STATUS_NAME(NVCV_ERROR_INVALID_ARGUMENT),
                                    MAKE_STATUS_NAME(NVCV_ERROR_INVALID_IMAGE_FORMAT),
                                    MAKE_STATUS_NAME(NVCV_ERROR_INVALID_OPERATION),
                                    MAKE_STATUS_NAME(NVCV_ERROR_DEVICE),
                                    MAKE_STATUS_NAME(NVCV_ERROR_NOT_READY),
                                    MAKE_STATUS_NAME(NVCV_ERROR_OUT_OF_MEMORY),
                                    MAKE_STATUS_NAME(NVCV_ERROR_INTERNAL),
                                    MAKE_STATUS_NAME(NVCV_ERROR_OVERFLOW),
                                    MAKE_STATUS_NAME(NVCV_ERROR_UNDERFLOW)));

// clang-format on

TEST_P(StatusNameTest, get_name)
{
    NVCVStatus  status = std::get<0>(GetParam());
    const char *gold   = std::get<1>(GetParam());

    EXPECT_STREQ(gold, nvcvStatusGetName(status));
}

TEST(StatusTest, main_thread_has_success_status_by_default)
{
    EXPECT_EQ(NVCV_SUCCESS, nvcvGetLastStatus());
    EXPECT_EQ(NVCV_SUCCESS, nvcvPeekAtLastStatus());
}

TEST(StatusTest, get_last_status_msg_success_has_correct_message)
{
    // resets status
    EXPECT_EQ(NVCV_SUCCESS, nvcvGetLastStatus());

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_SUCCESS, nvcvGetLastStatusMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}

TEST(StatusTest, peek_at_last_status_msg_success_has_correct_message)
{
    // resets status
    EXPECT_EQ(NVCV_SUCCESS, nvcvGetLastStatus());

    char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    ASSERT_EQ(NVCV_SUCCESS, nvcvPeekAtLastStatusMessage(msg, sizeof(msg)));
    EXPECT_STREQ("success", msg);
}
