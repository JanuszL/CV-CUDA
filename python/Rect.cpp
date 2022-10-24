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

#include "Rect.hpp"

#include "String.hpp"

#include <nvcv/Rect.h>

namespace nv::cvpy {

std::ostream &operator<<(std::ostream &out, const NVCVRectI &rc)
{
    return out << "RectI(x=" << rc.x << ",y=" << rc.y << ",width=" << rc.width << ",height=" << rc.height << ')';
}

void ExportRect(py::module &m)
{
    using namespace py::literals;

    py::class_<NVCVRectI>(m, "RectI")
        .def(py::init([]() { return NVCVRectI{}; }))
        .def(py::init(
                 [](int x, int y, int w, int h)
                 {
                     NVCVRectI r;
                     r.x      = x;
                     r.y      = y;
                     r.width  = w;
                     r.height = h;
                     return r;
                 }),
             "x"_a, "y"_a, "width"_a, "height"_a)
        .def_readwrite("x", &NVCVRectI::x)
        .def_readwrite("y", &NVCVRectI::y)
        .def_readwrite("width", &NVCVRectI::width)
        .def_readwrite("height", &NVCVRectI::height)
        .def("__repr__", &ToString<NVCVRectI>);
}

} // namespace nv::cvpy
