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

#include "CheckError.hpp"

#include "Exception.hpp"

#include <cuda_runtime.h>

#include <cstdarg>
#include <regex>

namespace nv::cv::util {

static std::string GetFunctionName(const char *stmt)
{
    static std::regex rgx("^([A-Za-z0-9_]+)\\(.*$");

    std::cmatch match;
    if (regex_match(stmt, match, rgx))
    {
        return match[1];
    }
    else
    {
        return "";
    }
}

namespace detail {
std::string GetCheckMessage()
{
    return "";
}

std::string GetCheckMessage(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    char buffer[NVCV_MAX_STATUS_MESSAGE_LENGTH];
    vsnprintf(buffer, sizeof(buffer) - 1, "%s: ", va);

    va_end(va);

    return buffer;
}
} // namespace detail

std::string FormatErrorMessage(const char *errname, const char *callstr, const std::string &msg)
{
    std::string funname = GetFunctionName(callstr);

    std::ostringstream ss;
    ss << '(';
    if (!funname.empty())
    {
        ss << funname << ':';
    }

    ss << errname << ')';
    if (!msg.empty())
    {
        ss << ' ' << msg;
    }

    return ss.str();
}

NVCVStatus TranslateError(cudaError_t err)
{
    switch (err)
    {
    case cudaErrorMemoryAllocation:
        return NVCV_ERROR_OUT_OF_MEMORY;

    case cudaErrorNotReady:
        return NVCV_ERROR_NOT_READY;

    case cudaErrorInvalidValue:
        return NVCV_ERROR_INVALID_ARGUMENT;

    default:
        return NVCV_ERROR_INTERNAL;
    }
}

void PreprocessError(cudaError_t err)
{
    // consume the error
    cudaGetLastError();
}

const char *ToString(cudaError_t err, const char **perrdescr)
{
    if (perrdescr != nullptr)
    {
        *perrdescr = cudaGetErrorString(err);
    }

    return cudaGetErrorName(err);
}

} // namespace nv::cv::util
