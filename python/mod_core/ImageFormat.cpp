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

#include "ImageFormat.hpp"

#include <nvcv/ImageFormat.hpp>

#include <sstream>

// So that pybind can export nv::cv::ImageFormat as python enum
namespace std {
template<>
struct underlying_type<nv::cv::ImageFormat>
{
    using type = uint64_t;
};
} // namespace std

namespace nv::cv {
size_t ComputeHash(const cv::ImageFormat &fmt)
{
    return std::hash<uint64_t>()(static_cast<uint64_t>(fmt));
}

} // namespace nv::cv

namespace nv::cvpy::priv {

static std::string ImageFormatToString(cv::ImageFormat fmt)
{
    const char *str = nvcvImageFormatGetName(fmt);

    const char *prefix = "NVCV_IMAGE_FORMAT_";

    std::ostringstream out;

    out << "nvcv.";

    if (strncmp(str, prefix, strlen(prefix)) == 0)
    {
        out << "Format." << str + strlen(prefix);
    }
    else
    {
        prefix = "ImageFormat";
        if (strncmp(str, prefix, strlen(prefix)) == 0)
        {
            out << "Format" << str + strlen(prefix);
        }
        else
        {
            out << "<Unknown image format: " << str << '>';
        }
    }

    return out.str();
}

void ExportImageFormat(py::module &m)
{
    py::enum_<cv::ImageFormat> fmt(m, "Format");

#define DEF(F)     fmt.value(#F, cv::FMT_##F);
// for formats that begin with a number, we must prepend it with underscore to make
// it a valid python identifier
#define DEF_NUM(F) fmt.value("_" #F, cv::FMT_##F);

#include "NVCVPythonImageFormatDefs.inc"

#undef DEF
#undef DEF_NUM

    fmt.export_values()
        .def_property_readonly("planes", &cv::ImageFormat::numPlanes)
        .def_property_readonly("channels", &cv::ImageFormat::numChannels);

    // Need to do this way because pybind11 doesn't allow enums to have methods.
    fmt.attr("__repr__") = py::cpp_function(&ImageFormatToString, py::name("__repr__"), py::is_method(fmt),
                                            "String representation of the image format.");
}

} // namespace nv::cvpy::priv
