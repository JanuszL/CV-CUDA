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

#include <nvcv/Version.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(nvcv, m)
{
    m.doc() = R"pbdoc(
        NVCV Python API reference
        ========================

        This is the Python API reference for the NVIDIA® NVCV library.
    )pbdoc";

    m.attr("__version__") = NVCV_VERSION_STRING;
}
