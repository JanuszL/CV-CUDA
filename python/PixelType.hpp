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

#ifndef NVCV_PYTHON_PIXELTYPE_HPP
#define NVCV_PYTHON_PIXELTYPE_HPP

#include <nvcv/PixelType.hpp>
#include <pybind11/pybind11.h>

namespace nv::cv {
size_t ComputeHash(const cv::PixelType &pix);
}

namespace nv::cvpy {
namespace py = pybind11;

void ExportPixelType(py::module &m);

} // namespace nv::cvpy

namespace pybind11::detail {

template<>
struct type_caster<nv::cv::PixelType>
{
    PYBIND11_TYPE_CASTER(nv::cv::PixelType, const_name("nvcv.Type"));

    bool          load(handle src, bool);
    static handle cast(nv::cv::PixelType type, return_value_policy /* policy */, handle /*parent */);
};

} // namespace pybind11::detail

#endif // NVCV_PYTHON_PIXELTYPE_HPP
