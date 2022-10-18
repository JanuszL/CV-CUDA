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

static std::string ProcessBufferInfoFormat(const std::string &fmt)
{
    // pybind11 (as of v2.6.2) doesn't recognize formats 'l' and 'L',
    // which according to https://docs.python.org/3/library/struct.html#format-characters
    // are equal to 'i' and 'I', respectively.
    if (fmt == "l")
    {
        return "i";
    }
    else if (fmt == "L")
    {
        return "I";
    }
    else
    {
        return fmt;
    }
}

py::dtype ToDType(const py::buffer_info &info)
{
    std::string fmt = ProcessBufferInfoFormat(info.format);

    PyObject *ptr = nullptr;
    if ((py::detail::npy_api::get().PyArray_DescrConverter_(py::str(fmt).ptr(), &ptr) == 0) || !ptr)
    {
        PyErr_Clear();
        return py::dtype(info);
    }
    else
    {
        return py::dtype(fmt);
    }
}

py::dtype ToDType(const std::string &fmt)
{
    py::buffer_info buf;
    buf.format = ProcessBufferInfoFormat(fmt);

    return ToDType(buf);
}

} // namespace nv::cvpy
