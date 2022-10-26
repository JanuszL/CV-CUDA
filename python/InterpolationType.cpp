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

#include "InterpolationType.hpp"

#include <operators/Types.h>

namespace nv::cvpy {

void ExportInterpolationType(py::module &m)
{
    py::enum_<NVCVInterpolationType>(m, "Interp")
        .value("NEAREST", NVCV_INTERP_NEAREST)
        .value("LINEAR", NVCV_INTERP_LINEAR)
        .value("CUBIC", NVCV_INTERP_CUBIC)
        .value("AREA", NVCV_INTERP_AREA);
}

} // namespace nv::cvpy
