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

#ifndef NVCV_PYTHON_PYUTIL_HPP
#define NVCV_PYTHON_PYUTIL_HPP

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <sstream>

namespace nv::cvpy {
namespace py = pybind11;

// Adds a method to an existing class
template<class T, class Func, class... Extra>
void DefClassMethod(const char *name, Func &&f, const Extra &...extra)
{
    py::type class_ = py::type::of<T>();

    // got from pt::class_<...>::def
    py::cpp_function cf(py::method_adaptor<py::type>(std::forward<Func>(f)), py::name(name), py::is_method(class_),
                        py::sibling(py::getattr(class_, name, py::none())), extra...);
    py::detail::add_class_method(class_, name, std::move(cf));
}

// Adds a static method to an existing class
template<class T, class Func, class... Extra>
void DefClassStaticMethod(const char *name, Func &&f, const Extra &...extra)
{
    py::type class_ = py::type::of<T>();

    // got from pt::class_<...>::def
    py::cpp_function cf(std::forward<Func>(f), py::name(name), py::scope(class_),
                        py::sibling(py::getattr(class_, name, py::none())), extra...);
    class_.attr(cf.name()) = py::staticmethod(cf);
}

void RegisterCleanup(py::module &m, std::function<void()> fn);

std::string GetFullyQualifiedName(py::handle h);

} // namespace nv::cvpy

#endif // NVCV_PYTHON_PYUTIL_HPP
