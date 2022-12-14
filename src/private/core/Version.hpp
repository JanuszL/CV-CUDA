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

#ifndef NVCV_PRIV_VERSION_HPP
#define NVCV_PRIV_VERSION_HPP

#include "Exception.hpp"

#include <nvcv/Version.h>

#include <cstdint>
#include <iosfwd>

// WAR for some unwanted macros
#include <sys/types.h>
#ifdef major
#    undef major
#endif
#ifdef minor
#    undef minor
#endif

namespace nv::cv::priv {

class Version
{
public:
    constexpr explicit Version(int major, int minor, int patch, int tweak = 0)
    {
        // Major can be > 99, the rest are limited to 99
        if (major < 0 || minor < 0 || minor > 99 || patch < 0 || patch > 99 || tweak < 0 || tweak > 99)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Invalid version code");
        }

        m_code = major * 1000000 + minor * 10000 + patch * 100 + tweak;
    }

    constexpr explicit Version(uint32_t versionCode)
        : m_code(versionCode)
    {
    }

    constexpr int major() const
    {
        return m_code / 1000000;
    }

    constexpr int minor() const
    {
        return (m_code % 1000000) / 10000;
    }

    constexpr int patch() const
    {
        return (m_code % 10000) / 100;
    }

    constexpr int tweak() const
    {
        return m_code % 100;
    }

    constexpr uint32_t code() const
    {
        return m_code;
    }

    constexpr auto operator<=>(const Version &that) const = default;

    // needs to be public so that type can be passed as non-type template parameter
    int m_code;
};

constexpr Version CURRENT_VERSION{NVCV_VERSION};

std::ostream &operator<<(std::ostream &out, const Version &ver);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_VERSION_HPP
