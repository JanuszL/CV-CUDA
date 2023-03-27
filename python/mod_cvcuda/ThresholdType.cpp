/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ThresholdType.hpp"

#include <cvcuda/Types.h>

namespace cvcudapy {

void ExportThresholdType(py::module &m)
{
    py::enum_<NVCVThresholdType>(m, "ThresholdType", py::arithmetic())
        .value("BINARY", NVCV_THRESH_BINARY)
        .value("BINARY_INV", NVCV_THRESH_BINARY_INV)
        .value("TRUNC", NVCV_THRESH_TRUNC)
        .value("TOZERO", NVCV_THRESH_TOZERO)
        .value("TOZERO_INV", NVCV_THRESH_TOZERO_INV)
        .value("OTSU", NVCV_THRESH_OTSU)
        .value("TRIANGLE", NVCV_THRESH_TRIANGLE);
}

} // namespace cvcudapy
