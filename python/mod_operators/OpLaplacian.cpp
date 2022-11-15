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

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <nvcv/operators/OpLaplacian.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
Tensor LaplacianInto(Tensor &output, Tensor &input, const int &ksize, const float &scale, NVCVBorderType border,
                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto laplacian = CreateOperator<cvop::Laplacian>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*laplacian});

    laplacian->submit(pstream->cudaHandle(), input, output, ksize, scale, border);

    return output;
}

Tensor Laplacian(Tensor &input, const int &ksize, const float &scale, NVCVBorderType border,
                 std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return LaplacianInto(output, input, ksize, scale, border, pstream);
}

ImageBatchVarShape LaplacianVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &ksize,
                                         Tensor &scale, NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto laplacian = CreateOperator<cvop::Laplacian>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, ksize, scale});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*laplacian});

    laplacian->submit(pstream->cudaHandle(), input, output, ksize, scale, border);

    return output;
}

ImageBatchVarShape LaplacianVarShape(ImageBatchVarShape &input, Tensor &ksize, Tensor &scale, NVCVBorderType border,
                                     std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input[i].format();
        cv::Size2D      size   = input[i].size();
        auto            image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return LaplacianVarShapeInto(output, input, ksize, scale, border, pstream);
}

} // namespace

void ExportOpLaplacian(py::module &m)
{
    using namespace pybind11::literals;

    m.def("laplacian", &Laplacian, "src"_a, "ksize"_a, "scale"_a = 1.f,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
    m.def("laplacian_into", &LaplacianInto, "dst"_a, "src"_a, "ksize"_a, "scale"_a = 1.f,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);

    m.def("laplacian", &LaplacianVarShape, "src"_a, "ksize"_a, "scale"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
    m.def("laplacian_into", &LaplacianVarShapeInto, "dst"_a, "src"_a, "ksize"_a, "scale"_a,
          "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
