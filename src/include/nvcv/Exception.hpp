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

namespace detail {
void ThrowException(NVCVStatus status);
}

/**
 * @defgroup NVCV_CPP_UTIL_EXCEPTION Exception
 * @{
*/

class Exception : public std::exception
{
public:
    explicit Exception(Status code, const char *msg = nullptr)
        : Exception(InternalCtorTag{}, code, msg)
    {
        nvcvSetThreadStatus(static_cast<NVCVStatus>(code), "%s", msg);
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

    friend void detail::ThrowException(NVCVStatus status);

    struct InternalCtorTag
    {
    };

    Exception(InternalCtorTag, Status code, const char *msg = nullptr)
        : m_code(code)
    {
        // Assuming
        snprintf(m_msgBuffer, sizeof(m_msgBuffer) - 1, "%s: %s", GetName(code), msg);

        m_msg = strchr(m_msgBuffer, ':');
        assert(m_msg != nullptr);
        m_msg += 2; // skip ': '
    }
};

inline void SetThreadError(std::exception_ptr e)
{
    try
    {
        if (e)
        {
            rethrow_exception(e);
        }
        else
        {
            nvcvSetThreadStatus(NVCV_SUCCESS, nullptr);
        }
    }
    catch (const Exception &e)
    {
        nvcvSetThreadStatus(static_cast<NVCVStatus>(e.code()), "%s", e.msg());
    }
    catch (const std::invalid_argument &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INVALID_ARGUMENT, "%s", e.what());
    }
    catch (const std::bad_alloc &)
    {
        nvcvSetThreadStatus(NVCV_ERROR_OUT_OF_MEMORY, "Not enough space for resource allocation");
    }
    catch (const std::exception &e)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "%s", e.what());
    }
    catch (...)
    {
        nvcvSetThreadStatus(NVCV_ERROR_INTERNAL, "Unexpected error");
    }
}

template<class F>
NVCVStatus ProtectCall(F &&fn)
{
    try
    {
        fn();
        return NVCV_SUCCESS;
    }
    catch (...)
    {
        SetThreadError(std::current_exception());
        return nvcvPeekAtLastError();
    }
}

/**@}*/

}} // namespace nv::cv

#endif // NVCV_EXCEPTION_HPP
