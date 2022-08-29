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
#include "Exception.hpp"

#include <driver_types.h> // for cudaError
#include <nvcv/Status.h>

#include <cstring>
#include <iostream>
#include <string>

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
std::string GetCheckMessage();
std::string GetCheckMessage(const char *fmt, ...);
} // namespace detail

std::string FormatErrorMessage(const char *errname, const char *callstr, const std::string &msg);

// CUDA -----------------------

inline bool CheckSucceeded(cudaError_t err)
{
    return err == cudaSuccess;
}

NVCVStatus  TranslateError(cudaError_t err);
const char *ToString(cudaError_t err, const char **perrdescr = nullptr);
void        PreprocessError(cudaError_t err);

// Driver functions --------------------

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

template<class T>
void CheckThrow(T error, const char *file, int line, const char *stmt, const std::string &errmsg)
{
    (void)file;
    (void)line;

    PreprocessError(error);

    if (!CheckSucceeded(error))
    {
        throw Exception(TranslateError(error), "%s", FormatErrorMessage(ToString(error), stmt, errmsg).c_str());
    }
}

template<class T>
bool CheckLog(T error, const char *file, int line, const char *stmt, const std::string &errmsg)
{
    PreprocessError(error);
    if (!CheckSucceeded(error))
    {
        // TODO: replace with a real log facility
        std::cerr
#if NVCV_EXPOSE_CODE
            << file << ":" << line << ' '
#endif
            << FormatErrorMessage(ToString(error), stmt, errmsg);
        return false;
    }
    return true;
}

#define NVCV_CHECK_THROW(STMT, ...)                                                        \
    [&]()                                                                                  \
    {                                                                                      \
        ::nv::cv::util::CheckThrow((STMT), NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO, \
                                   NVCV_OPTIONAL_STRINGIFY(STMT),                          \
                                   ::nv::cv::util::detail::GetCheckMessage(__VA_ARGS__));  \
    }()

#define NVCV_CHECK_LOG(STMT, ...)                                                               \
    [&]()                                                                                       \
    {                                                                                           \
        return ::nv::cv::util::CheckLog((STMT), NVCV_SOURCE_FILE_NAME, NVCV_SOURCE_FILE_LINENO, \
                                        NVCV_OPTIONAL_STRINGIFY(STMT),                          \
                                        ::nv::cv::util::detail::GetCheckMessage(__VA_ARGS__));  \
    }()

} // namespace nv::cv::util

#endif // NVCV_UTIL_CHECK_ERROR_HPP
