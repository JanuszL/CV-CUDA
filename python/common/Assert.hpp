/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_PYTHON_ASSERT_HPP
#define NVCV_PYTHON_ASSERT_HPP

#include <nvcv/Status.h>

#include <functional>

namespace nv::cvpy::util {

#ifdef _MSC_VER
#    define NVCV_NORETURN __declspec(noreturn)
#else
#    define NVCV_NORETURN __attribute__((noreturn))
#endif

NVCV_NORETURN void DoAssert(const char *file, int line, const char *cond);

#if !defined(NVCV_DEBUG)
#    define NVCV_DEBUG (!NDEBUG)
#endif

#if NVCV_EXPOSE_CODE
#    define NVCV_SOURCE_FILE_NAME      __FILE__
#    define NVCV_SOURCE_FILE_LINENO    __LINE__
#    define NVCV_OPTIONAL_STRINGIFY(X) #    X
#else
#    define NVCV_SOURCE_FILE_NAME      ""
#    define NVCV_SOURCE_FILE_LINENO    0
#    define NVCV_OPTIONAL_STRINGIFY(X) ""
#endif

// allows overriding of NVCV_ASSERT definition
#if !defined(NVCV_ASSERT)
#    if NVCV_DEBUG
#        define NVCV_ASSERT(x)                                                               \
            do                                                                               \
            {                                                                                \
                if (!(x))                                                                    \
                {                                                                            \
                    nv::cvpy::util::DoAssert(NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO, \
                                             NVCV_OPTIONAL_STRINGIFY(x));                    \
                }                                                                            \
            }                                                                                \
            while (1 == 0)
#    else
#        define NVCV_ASSERT(x) \
            do                 \
            {                  \
            }                  \
            while (1 == 0)
#    endif
#endif // !defined(NVCV_ASSERT)

} // namespace nv::cvpy::util

#endif // NVCV_PYTHON_ASSERT_HPP
