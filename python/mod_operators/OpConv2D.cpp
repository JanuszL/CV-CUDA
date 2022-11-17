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
#include <nvcv/operators/OpConv2D.hpp>
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

ImageBatchVarShape Conv2DVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output, ImageBatchVarShape &kernel,
                                      Tensor &kernel_anchor, NVCVBorderType border, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto conv2D = CreateOperator<cvop::Conv2D>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, kernel, kernel_anchor});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*conv2D});

    conv2D->submit(pstream->cudaHandle(), input, output, kernel, kernel_anchor, border);

    return output;
}

ImageBatchVarShape Conv2DVarShape(ImageBatchVarShape &input, ImageBatchVarShape &kernel, Tensor &kernel_anchor,
                                  NVCVBorderType border, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input[i].format();
        cv::Size2D      size   = input[i].size();
        auto            image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return Conv2DVarShapeInto(input, output, kernel, kernel_anchor, border, pstream);
}

} // namespace

void ExportOpConv2D(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::ImageBatchVarShape>("conv2d", &Conv2DVarShape, "kernel"_a, "kernel_anchor"_a,
                                                   "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
                                                   "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("conv2d_into", &Conv2DVarShapeInto, "output"_a, "kernel"_a,
                                                   "kernel_anchor"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
                                                   py::kw_only(), "stream"_a     = nullptr);
}

} // namespace nv::cvpy
