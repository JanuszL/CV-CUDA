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

/**
 * @file Exception.hpp
 *
 * @brief Declaration of NVCV C++ exception classes.
 */

#ifndef NVCV_EXCEPTION_HPP
#define NVCV_EXCEPTION_HPP

#include <nvcv/Status.hpp>

#include <cassert>
#include <cstring>

namespace nv { namespace cv {

class Exception : public std::exception
{
public:
    explicit Exception(Status code, const char *msg = nullptr)
        : m_code(code)
    {
        // Assuming
        snprintf(m_msgBuffer, sizeof(m_msgBuffer) - 1, "%s: %s", GetName(code), msg);

        m_msg = strchr(m_msgBuffer, ':');
        assert(m_msg != nullptr);
        m_msg += 2; // skip ': '
    }

    Status code() const
    {
        return m_code;
    }

    const char *msg() const
    {
        return m_msg;
    }

    const char *what() const noexcept override
    {
        return m_msgBuffer;
    }

private:
    Status      m_code;
    const char *m_msg;

    // 64: maximum size of string representatio of a status enum
    // 2: ': '
    char m_msgBuffer[NVCV_MAX_STATUS_MESSAGE_LENGTH + 64 + 2];
};

}} // namespace nv::cv

#endif // NVCV_EXCEPTION_HPP
