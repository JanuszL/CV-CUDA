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

#include <gtest/gtest.h>
#include <util/Exception.hpp>

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
    NVCVStatus last = nvcvPeekAtLastStatusMessage(detail, sizeof(detail));

    if (last != NVCV_SUCCESS && (last == lhs || last == rhs))
    {
        res << "\n  Detail: " << detail;
    }

    return res;
}
#endif

#define NVCV_DETAIL_THROW_STATUS(FAIL_KIND, CODE, STMT)                                              \
    do                                                                                               \
    {                                                                                                \
        NVCVStatus status = (STMT);                                                                  \
        if (status == (CODE))                                                                        \
        {                                                                                            \
            SUCCEEDED();                                                                             \
        }                                                                                            \
        else                                                                                         \
        {                                                                                            \
            FAIL_KIND() << "Call to " #STMT << " expected to fail with return status code  " << CODE \
                        << ", but returned instead " << status;                                      \
        }                                                                                            \
    }                                                                                                \
    while (false)

#define NVCV_DETAIL_RETURN_STATUS(FAIL_KIND, CODE, STMT) \
    STMT;                                                \
    FAIL_KIND() << "Call to " #STMT << " expected have thrown an exception with code " << (CODE)

#define NVCV_DETAIL_CHECK_STATUS(TYPE, FAIL_KIND, CODE, STMT)                                                          \
    try                                                                                                                \
    {                                                                                                                  \
        TYPE;                                                                                                          \
    }                                                                                                                  \
    catch (::nv::cv::util::Exception & e)                                                                              \
    {                                                                                                                  \
        if ((CODE) == e.code())                                                                                        \
        {                                                                                                              \
            SUCCEED();                                                                                                 \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            FAIL_KIND() << "Call to " #STMT << " expected have thrown an exception with code " << (CODE)               \
                        << ", but it's '" << e.what() << "' instead";                                                  \
        }                                                                                                              \
    }                                                                                                                  \
    catch (std::exception & e)                                                                                         \
    {                                                                                                                  \
        FAIL_KIND() << "Call to " #STMT << " expected have thrown an exception with code " << (CODE) << ", but threw " \
                    << typeid(e).name() << " with message '" << e.what() << "'";                                       \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        FAIL_KIND() << "Call to " #STMT << " expected have thrown an exception with code " << (CODE)                   \
                    << ", but threw an unknown exception";                                                             \
    }

#define NVCV_ASSERT_THROW_STATUS(CODE, STMT) \
    NVCV_DETAIL_CHECK_STATUS(NVCV_DETAIL_THROW_STATUS(FAIL, CODE, STMT), FAIL, CODE, STMT)

#define NVCV_EXPECT_THROW_STATUS(CODE, STMT) \
    NVCV_DETAIL_CHECK_STATUS(NVCV_DETAIL_THROW_STATUS(ADD_FAILURE, CODE, STMT), ADD_FAILURE, CODE, STMT)

#define NVCV_ASSERT_STATUS(CODE, STMT) \
    NVCV_DETAIL_CHECK_STATUS(NVCV_DETAIL_RETURN_STATUS(FAIL, CODE, STMT), FAIL, CODE, STMT)

#define NVCV_EXPECT_STATUS(CODE, STMT) \
    NVCV_DETAIL_CHECK_STATUS(NVCV_DETAIL_RETURN_STATUS(ADD_FAILURE, CODE, STMT), ADD_FAILURE, CODE, STMT)

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
