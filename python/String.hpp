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

#ifndef NVCV_PYTHON_STRING_HPP
#define NVCV_PYTHON_STRING_HPP

#include <sstream>
#include <string>

namespace nv::cvpy {

std::string FormatString(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

// Make it easier to use ostreams to define __repr__
template<class T>
std::string ToString(const T &v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

} // namespace nv::cvpy

#endif // NVCV_PYTHON_STRING_HPP
