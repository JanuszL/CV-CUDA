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

#include "BorderType.hpp"
#include "ColorConversionCode.hpp"
#include "InterpolationType.hpp"
#include "MorphologyType.hpp"
#include "Operators.hpp"

#include <nvcv/operators/Version.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(cvcuda, m)
{
    // Automatically import nvcv module, needed by operators
    py::module_::import("nvcv");

    m.doc() = R"pbdoc(
        CV-CUDA Python API reference
        ========================

        This is the Python API reference for the NVIDIA® CV-CUDA library.
    )pbdoc";

    m.attr("__version__") = NVCV_OP_VERSION_STRING;

    using namespace cvcudapy;

    // Operators' auxiliary entities
    ExportInterpolationType(m);
    ExportBorderType(m);
    ExportMorphologyType(m);
    ExportColorConversionCode(m);

    // Operators
    ExportOpReformat(m);
    ExportOpResize(m);
    ExportOpCustomCrop(m);
    ExportOpNormalize(m);
    ExportOpConvertTo(m);
    ExportOpPadAndStack(m);
    ExportOpCopyMakeBorder(m);
    ExportOpRotate(m);
    ExportOpErase(m);
    ExportOpGaussian(m);
    ExportOpMedianBlur(m);
    ExportOpLaplacian(m);
    ExportOpAverageBlur(m);
    ExportOpConv2D(m);
    ExportOpBilateralFilter(m);
    ExportOpCenterCrop(m);
    ExportOpWarpAffine(m);
    ExportOpWarpPerspective(m);
    ExportOpChannelReorder(m);
    ExportOpMorphology(m);
    ExportOpFlip(m);
    ExportOpCvtColor(m);
    ExportOpComposite(m);
    ExportOpGammaContrast(m);
    ExportOpPillowResize(m);
}
