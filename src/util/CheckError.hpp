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

#ifndef NVCV_UTIL_CHECK_ERROR_HPP
#define NVCV_UTIL_CHECK_ERROR_HPP

#include "Assert.h"

#include <driver_types.h> // for cudaError
#include <nvcv/Status.h>

#include <cstring>
#include <iostream>
#include <string>
#include <string_view>

#if NVCV_EXPORTING
#    include <private/core/Exception.hpp>
#else
#    include <nvcv/Exception.hpp>
#endif

// Here we define the CHECK_ERROR macro that converts error values into exceptions
// or log messages. It can be extended to other errors. At minimum, you need to define:
// * inline bool CheckSucceeded(ErrorType err)
// * const char *ToString(NvError err, const char **perrdescr=nullptr);
//
// Optionally, you can define:
// * NVCVStatus TranslateError(NvError err);
//   by default it translates success to NVCV_SUCCESS, failure to NVCV_ERROR_INTERNAL
// * void PreprocessError(T err);
//   If you need to swallow up the error, CUDA needs that. Or any other kind of
//   processing when the error occurs

namespace nv::cv::util {

namespace detail {
const char *GetCheckMessage(char *buf, int buflen);
char       *GetCheckMessage(char *buf, int buflen, const char *fmt, ...);
std::string FormatErrorMessage(const std::string_view &errname, const std::string_view &callstr,
                               const std::string_view &msg);
} // namespace detail

// CUDA -----------------------

inline bool CheckSucceeded(cudaError_t err)
{
    return err == cudaSuccess;
}

NVCVStatus  TranslateError(cudaError_t err);
const char *ToString(cudaError_t err, const char **perrdescr = nullptr);
void        PreprocessError(cudaError_t err);

// Default implementation --------------------

template<class T>
NVCVStatus TranslateError(T err)
{
    if (CheckSucceeded(err))
    {
        return NVCV_SUCCESS;
    }
    else
    {
        return NVCV_ERROR_INTERNAL;
    }
}

template<class T>
inline void PreprocessError(T err)
{
}

namespace detail {

template<class T>
void DoThrow(T error, const char *file, int line, const std::string_view &stmt, const std::string_view &errmsg)
{
#if NVCV_EXPORTING
    using cv::priv::Exception;
#else
    using cv::Exception;
#endif

    // Can we expose source file data?
    if (file != nullptr)
    {
        throw Exception(TranslateError(error), "%s:%d %s", file, line,
                        FormatErrorMessage(ToString(error), stmt, errmsg).c_str());
    }
    else
    {
        throw Exception(TranslateError(error), "%s", FormatErrorMessage(ToString(error), stmt, errmsg).c_str());
    }
}

template<class T>
void DoLog(T error, const char *file, int line, const std::string_view &stmt, const std::string_view &errmsg)
{
    // TODO: replace with a real log facility

    // Can we expose source file data?
    if (file != nullptr)
    {
        std::cerr << file << ":" << line << ' ';
    }
    std::cerr << FormatErrorMessage(ToString(error), stmt, errmsg);
}

} // namespace detail

#define NVCV_CHECK_THROW(STMT, ...)                                                                                    \
    [&]()                                                                                                              \
    {                                                                                                                  \
        using ::nv::cv::util::PreprocessError;                                                                         \
        using ::nv::cv::util::CheckSucceeded;                                                                          \
        auto status = (STMT);                                                                                          \
        PreprocessError(status);                                                                                       \
        if (!CheckSucceeded(status))                                                                                   \
        {                                                                                                              \
            char buf[NVCV_MAX_STATUS_MESSAGE_LENGTH];                                                                  \
            ::nv::cv::util::detail::DoThrow(status, NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO,                    \
                                            NVCV_OPTIONAL_STRINGIFY(STMT),                                             \
                                            ::nv::cv::util::detail::GetCheckMessage(buf, sizeof(buf), ##__VA_ARGS__)); \
        }                                                                                                              \
    }()

#define NVCV_CHECK_LOG(STMT, ...)                                                                                    \
    [&]()                                                                                                            \
    {                                                                                                                \
        using ::nv::cv::util::PreprocessError;                                                                       \
        using ::nv::cv::util::CheckSucceeded;                                                                        \
        auto status = (STMT);                                                                                        \
        PreprocessError(status);                                                                                     \
        if (!CheckSucceeded(status))                                                                                 \
        {                                                                                                            \
            char buf[NVCV_MAX_STATUS_MESSAGE_LENGTH];                                                                \
            ::nv::cv::util::detail::DoLog(status, NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO,                    \
                                          NVCV_OPTIONAL_STRINGIFY(STMT),                                             \
                                          ::nv::cv::util::detail::GetCheckMessage(buf, sizeof(buf), ##__VA_ARGS__)); \
            return false;                                                                                            \
        }                                                                                                            \
        else                                                                                                         \
        {                                                                                                            \
            return true;                                                                                             \
        }                                                                                                            \
    }()

} // namespace nv::cv::util

#endif // NVCV_UTIL_CHECK_ERROR_HPP
