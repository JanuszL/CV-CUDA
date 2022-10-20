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

#include "PyUtil.hpp"

#include "Assert.hpp"
#include "String.hpp"

#include <cstdlib>

namespace nv::cvpy {

void RegisterCleanup(py::module &m, std::function<void()> fn)
{
    // Functions registered with python's atexit will be executed after
    // the script ends, but before python interpreter is torn down. That's
    // a good place for our cleanup.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function(std::move(fn)));
}

std::string GetFullyQualifiedName(py::handle h)
{
    py::handle type = h.get_type();

    std::ostringstream ss;
    ss << type.attr("__module__").cast<std::string>() << '.' << type.attr("__qualname__").cast<std::string>();
    return ss.str();
}
} // namespace nv::cvpy
