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

#include <common/Printers.hpp>
#include <gtest/gtest.h>
#include <util/Exception.hpp>

#include <type_traits>

#if NVCV_SYSTEM_TESTS
// Let's hijack gtest and create an overload for NVCVStatus that
// prints out the status message. This will end up being called by
// ASSERT_EQ / EXPECT_EQ.
inline ::testing::AssertionResult CmpHelperEQFailure(const char *lhs_expression, const char *rhs_expression,
                                                     NVCVStatus lhs, NVCVStatus rhs)
{
    using namespace ::testing::internal;

    auto res = EqFailure(lhs_expression, rhs_expression, FormatForComparisonFailureMessage(lhs, rhs),
                         FormatForComparisonFailureMessage(rhs, lhs), false);

    char       detail[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    NVCVStatus last = nvcvPeekAtLastErrorMessage(detail, sizeof(detail));

    if (last != NVCV_SUCCESS && (last == lhs || last == rhs))
    {
        res << "\n  Detail: " << detail;
    }

    return res;
}
#endif

#define NVCV_DETAIL_CHECK_STATUS(FAIL_KIND, STATUS, ...)                                                     \
    try                                                                                                      \
    {                                                                                                        \
        auto fn = [&]<class T>(T) -> NVCVStatus                                                              \
        {                                                                                                    \
            if constexpr (sizeof(T) != 0 && std::is_same_v<decltype(__VA_ARGS__), NVCVStatus>)               \
            {                                                                                                \
                return __VA_ARGS__;                                                                          \
            }                                                                                                \
            else                                                                                             \
            {                                                                                                \
                __VA_ARGS__;                                                                                 \
                return NVCV_SUCCESS;                                                                         \
            }                                                                                                \
        };                                                                                                   \
        NVCVStatus status = fn(0);                                                                           \
        if (status != (STATUS))                                                                              \
        {                                                                                                    \
            if ((STATUS) == NVCV_SUCCESS)                                                                    \
            {                                                                                                \
                FAIL_KIND() << "Call to " #__VA_ARGS__ << " expected to succeed"                             \
                            << ", but failed with status " << status;                                        \
            }                                                                                                \
            else                                                                                             \
            {                                                                                                \
                FAIL_KIND() << "Call to " #__VA_ARGS__ << " expected to fail with status  " << STATUS        \
                            << ", but succeeded";                                                            \
            }                                                                                                \
        }                                                                                                    \
    }                                                                                                        \
    catch (::nv::cv::util::Exception & e)                                                                    \
    {                                                                                                        \
        if ((STATUS) == e.code())                                                                            \
        {                                                                                                    \
            SUCCEED();                                                                                       \
        }                                                                                                    \
        else                                                                                                 \
        {                                                                                                    \
            FAIL_KIND() << "Call to '" #__VA_ARGS__ "' expected to fail with status " << (STATUS)            \
                        << ", but failed with '" << e.what() << "' instead";                                 \
        }                                                                                                    \
    }                                                                                                        \
    catch (std::exception & e)                                                                               \
    {                                                                                                        \
        FAIL_KIND() << "Call to '" #__VA_ARGS__ "' expected fail with status " << (STATUS) << ", but threw " \
                    << typeid(e).name() << " with message '" << e.what() << "'";                             \
    }                                                                                                        \
    catch (...)                                                                                              \
    {                                                                                                        \
        FAIL_KIND() << "Call to '" #__VA_ARGS__ "' expected fail with status " << (STATUS)                   \
                    << ", but threw an unknown exception";                                                   \
    }

#define NVCV_ASSERT_STATUS(STATUS, ...) NVCV_DETAIL_CHECK_STATUS(FAIL, STATUS, __VA_ARGS__)

#define NVCV_EXPECT_STATUS(STATUS, ...) NVCV_DETAIL_CHECK_STATUS(ADD_FAILURE, STATUS, __VA_ARGS__)

#define NVCV_ASSERT_THROW(E, ...)                                                                              \
    try                                                                                                        \
    {                                                                                                          \
        __VA_ARGS__;                                                                                           \
        ADD_FAILURE() << "Expected an exception of type " #E ", got none";                                     \
    }                                                                                                          \
    catch (E & e)                                                                                              \
    {                                                                                                          \
    }                                                                                                          \
    catch (std::exception & e)                                                                                 \
    {                                                                                                          \
        ADD_FAILURE() << "Expected an exception of type " #E ", got " << typeid(e).name() << " with message '" \
                      << e.what() << "'";                                                                      \
    }                                                                                                          \
    catch (...)                                                                                                \
    {                                                                                                          \
        ADD_FAILURE() << "Expected an exception of type " #E ", got an unknown exception";                     \
    }
