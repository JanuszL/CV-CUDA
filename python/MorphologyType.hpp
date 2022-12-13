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

#ifndef NVCV_PYTHON_MORPHOLOGY_TYPE_HPP
#define NVCV_PYTHON_MORPHOLOGY_TYPE_HPP

#include <pybind11/pybind11.h>

namespace nv::cvpy {
namespace py = ::pybind11;

void ExportMorphologyType(py::module &m);

} // namespace nv::cvpy

#endif // NVCV_PYTHON_MORPHOLOGY_TYPE_HPP
