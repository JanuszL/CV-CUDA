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

#include "String.hpp"

#include <cstdarg>
#include <cstdio>

namespace nv::cvpy {

std::string FormatString(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer) - 1, fmt, va);
    buffer[sizeof(buffer) - 1] = '\0'; // better be safe against truncation

    va_end(va);

    return buffer;
}

} // namespace nv::cvpy
