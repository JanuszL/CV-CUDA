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
#include <mod_core/Image.hpp>
#include <mod_core/ImageBatch.hpp>
#include <mod_core/ResourceGuard.hpp>
#include <mod_core/Stream.hpp>
#include <mod_core/Tensor.hpp>
#include <nvcv/operators/OpConv2D.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {

std::shared_ptr<ImageBatchVarShape> Conv2DVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                       ImageBatchVarShape &kernel, Tensor &kernel_anchor,
                                                       NVCVBorderType border, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto conv2D = CreateOperator<cvop::Conv2D>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, kernel, kernel_anchor});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*conv2D});

    conv2D->submit(pstream->handle(), input.impl(), output.impl(), kernel.impl(), kernel_anchor.impl(), border);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> Conv2DVarShape(ImageBatchVarShape &input, ImageBatchVarShape &kernel,
                                                   Tensor &kernel_anchor, NVCVBorderType border,
                                                   std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return Conv2DVarShapeInto(input, *output, kernel, kernel_anchor, border, pstream);
}

} // namespace

void ExportOpConv2D(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<ImageBatchVarShape>("conv2d", &Conv2DVarShape, "kernel"_a, "kernel_anchor"_a,
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
                                       "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("conv2d_into", &Conv2DVarShapeInto, "output"_a, "kernel"_a, "kernel_anchor"_a,
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
