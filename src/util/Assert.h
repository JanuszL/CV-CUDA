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

#ifndef NVCV_UTIL_ASSERT_H
#define NVCV_UTIL_ASSERT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
#    define NVCV_ASSERT_NORETURN [[noreturn]]
{
#else
#    define NVCV_ASSERT_NORETURN __attribute__((noreturn))
#endif

#ifdef NVCV_ASSERT_OVERRIDE_HEADER
#    include NVCV_ASSERT_OVERRIDE_HEADER
#endif

NVCV_ASSERT_NORETURN void NvCVAssert(const char *file, int line, const char *cond);

#if !defined(NVCV_DEBUG)
#    define NVCV_DEBUG (!NDEBUG)
#endif

#if NVCV_EXPOSE_CODE
#    define NVCV_SOURCE_FILE_NAME      __FILE__
#    define NVCV_SOURCE_FILE_LINENO    __LINE__
#    define NVCV_OPTIONAL_STRINGIFY(X) #    X
#else
#    define NVCV_SOURCE_FILE_NAME      NULL
#    define NVCV_SOURCE_FILE_LINENO    0
#    define NVCV_OPTIONAL_STRINGIFY(X) ""
#endif

// allows overriding of NVCV_ASSERT definition
#if !defined(NVCV_ASSERT)
#    if NVCV_DEBUG
#        define NVCV_ASSERT(x)                                                                              \
            do                                                                                              \
            {                                                                                               \
                if (!(x))                                                                                   \
                {                                                                                           \
                    NvCVAssert(NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO, NVCV_OPTIONAL_STRINGIFY(x)); \
                }                                                                                           \
            }                                                                                               \
            while (1 == 0)
#    else
#        define NVCV_ASSERT(x) \
            do                 \
            {                  \
            }                  \
            while (1 == 0)
#    endif
#endif // !defined(NVCV_ASSERT)

#ifdef __cplusplus
}
#endif

#endif // NVCV_UTIL_ASSERT_H
