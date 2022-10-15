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

namespace nv::cvpy {

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

} // namespace nv::cvpy
