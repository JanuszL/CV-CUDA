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
#include <nvcv/operators/OpResize.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
Tensor ResizeInto(Tensor &input, Tensor &output, NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvop::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*resize});

    resize->submit(pstream->cudaHandle(), input, output, interp);

    return std::move(output);
}

Tensor Resize(Tensor &input, const Shape &out_shape, NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    Tensor output
        = Tensor::Create(cv::TensorShape(out_shape.data(), out_shape.size(), input.shape().layout()), input.dtype());

    return ResizeInto(input, output, interp, pstream);
}

ImageBatchVarShape ResizeVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                      NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto resize = CreateOperator<cvop::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*resize});

    resize->submit(pstream->cudaHandle(), input, output, interp);

    return output;
}

ImageBatchVarShape ResizeVarShape(ImageBatchVarShape &input, const std::vector<std::tuple<int, int>> &out_size,
                                  NVCVInterpolationType interp, std::optional<Stream> pstream)
{
    if (input.numImages() != (int)out_size.size())
    {
        throw std::runtime_error("Number of input images must be equal to the number of elements in output size list ");
    }

    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input[i].format();
        auto            size   = out_size[i];
        auto            image  = Image::Create({std::get<0>(size), std::get<1>(size)}, format);
        output.pushBack(image);
    }

    return ResizeVarShapeInto(input, output, interp, pstream);
}

} // namespace

void ExportOpResize(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::Tensor>("resize", &Resize, "shape"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
                                       "stream"_a = nullptr);
    util::DefClassMethod<priv::Tensor>("resize_into", &ResizeInto, "out"_a, "interp"_a = NVCV_INTERP_LINEAR,
                                       py::kw_only(), "stream"_a = nullptr);

    util::DefClassMethod<priv::ImageBatchVarShape>(
        "resize", &ResizeVarShape, "sizes"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(), "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("resize_into", &ResizeVarShapeInto, "out"_a,
                                                   "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
                                                   "stream"_a = nullptr);
}

} // namespace nv::cvpy
