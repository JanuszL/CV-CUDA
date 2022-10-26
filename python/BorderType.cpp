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

#include "BorderType.hpp"

#include <operators/Types.h>

namespace nv::cvpy {

void ExportBorderType(py::module &m)
{
    py::enum_<NVCVBorderType>(m, "Border")
        .value("CONSTANT", NVCV_BORDER_CONSTANT)
        .value("REPLICATE", NVCV_BORDER_REPLICATE)
        .value("REFLECT", NVCV_BORDER_REFLECT)
        .value("WRAP", NVCV_BORDER_WRAP)
        .value("REFLECT101", NVCV_BORDER_REFLECT101);
}

} // namespace nv::cvpy
