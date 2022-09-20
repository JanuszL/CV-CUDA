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

#ifndef NVCV_UTIL_EXCEPTION_HPP
#define NVCV_UTIL_EXCEPTION_HPP

#include <nvcv/Status.h>

#include <cstring>

#ifdef __GNUC__
#    undef __DEPRECATED
#endif
#include <strstream>

namespace nv::cv::util {

class Exception : public std::exception
{
public:
    explicit Exception(NVCVStatus code, const char *fmt, ...)
#if __GNUC__
        // first argument is actually 'this'
        __attribute__((format(printf, 3, 4)));
#else
        ;
#endif

    explicit Exception(NVCVStatus code);

    NVCVStatus  code() const;
    const char *msg() const;

    const char *what() const noexcept override;

    template<class T>
    Exception &&operator<<(const T &v) &&
    {
        // TODO: must avoid allocating memory from heap, can't use ostringstream
        std::ostream ss(&m_strbuf);
        ss << v << std::flush;
        return std::move(*this);
    }

private:
    NVCVStatus m_code;
    char       m_buffer[NVCV_MAX_STATUS_MESSAGE_LENGTH + 64 + 2];

    class StrBuffer : public std::strstreambuf
    {
    public:
        using std::strstreambuf::seekpos;
        using std::strstreambuf::strstreambuf;
    };

    StrBuffer m_strbuf;
};

} // namespace nv::cv::util

#endif // NVCV_UTIL_EXCEPTION_HPP
