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

PYBIND11_MODULE(nvcv_operators, m)
{
    m.attr("__name__") = "nvcv.operators";

    auto nvcv = py::module_::import("nvcv");

    m.doc() = R"pbdoc(
        NVCV Operators Python API reference
        ========================

        This is the Python API reference for the NVIDIA® NVCVOperators  library.
    )pbdoc";

    m.attr("__version__") = NVCV_OP_VERSION_STRING;

    using namespace nv::cvpy;

    // Operators' auxiliary entities
    ExportInterpolationType(nvcv);
    ExportBorderType(nvcv);
    ExportMorphologyType(nvcv);
    ExportColorConversionCode(nvcv);

    // Operators
    ExportOpReformat(nvcv);
    ExportOpResize(nvcv);
    ExportOpCustomCrop(nvcv);
    ExportOpNormalize(nvcv);
    ExportOpConvertTo(nvcv);
    ExportOpPadAndStack(nvcv);
    ExportOpCopyMakeBorder(nvcv);
    ExportOpRotate(nvcv);
    ExportOpErase(nvcv);
    ExportOpGaussian(nvcv);
    ExportOpMedianBlur(nvcv);
    ExportOpLaplacian(nvcv);
    ExportOpAverageBlur(nvcv);
    ExportOpConv2D(nvcv);
    ExportOpBilateralFilter(nvcv);
    ExportOpCenterCrop(nvcv);
    ExportOpWarpAffine(nvcv);
    ExportOpWarpPerspective(nvcv);
    ExportOpChannelReorder(nvcv);
    ExportOpMorphology(nvcv);
    ExportOpFlip(nvcv);
    ExportOpCvtColor(nvcv);
    ExportOpComposite(nvcv);
    ExportOpGammaContrast(nvcv);
    ExportOpPillowResize(nvcv);
}
