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
#include "Cache.hpp"
#include "Container.hpp"
#include "CudaBuffer.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "ImageFormat.hpp"
#include "InterpolationType.hpp"
#include "Operators.hpp"
#include "PixelType.hpp"
#include "Rect.hpp"
#include "Resource.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

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

    using namespace nv::cvpy;

    // Core entities
    Cache::Export(m);

    {
        py::module_ cuda = m.def_submodule("cuda");
        Stream::Export(cuda);
        CudaBuffer::Export(cuda);
    }

    ExportImageFormat(m);
    ExportPixelType(m);
    ExportRect(m);
    ExportBorderType(m);
    Resource::Export(m);
    Container::Export(m);
    Tensor::Export(m);
    Image::Export(m);
    ImageBatchVarShape::Export(m);

    // Operators' auxiliary entities
    ExportInterpolationType(m);

    // Operators
    ExportOpReformat(m);
    ExportOpResize(m);
    ExportOpCustomCrop(m);
    ExportOpNormalize(m);
    ExportOpConvertTo(m);
    ExportOpPadAndStack(m);
}
