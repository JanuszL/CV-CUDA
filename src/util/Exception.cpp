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

#include "Exception.hpp"

#include "Assert.h"
#include "Status.hpp"

#include <cstdarg>

namespace nv::cv::util {

Exception::Exception(NVCVStatus code)
    : Exception(code, "%s", "")
{
}

Exception::Exception(NVCVStatus code, const char *fmt, ...)
    : m_code(code)
    , m_strbuf{m_buffer, sizeof(m_buffer), m_buffer}
{
    va_list va;
    va_start(va, fmt);

    snprintf(m_buffer, sizeof(m_buffer) - 1, "%s: ", ToString(code));

    size_t len = strlen(m_buffer);
    vsnprintf(m_buffer + len, sizeof(m_buffer) - len - 1, fmt, va);

    va_end(va);

    // Next character written will be appended to m_buffer
    m_strbuf.seekpos(strlen(m_buffer), std::ios_base::out);
}

NVCVStatus Exception::code() const
{
    return m_code;
}

const char *Exception::msg() const
{
    // Only return the message part
    const char *out = strchr(m_buffer, ':');
    NVCV_ASSERT(out != nullptr);

    return out += 2; // skip ': '
}

const char *Exception::what() const noexcept
{
    return m_buffer;
}

} // namespace nv::cv::util
