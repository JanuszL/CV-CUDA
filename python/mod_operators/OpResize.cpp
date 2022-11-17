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
#include <mod_core/Image.hpp>
#include <mod_core/ImageBatch.hpp>
#include <mod_core/ResourceGuard.hpp>
#include <mod_core/Stream.hpp>
#include <mod_core/Tensor.hpp>
#include <nvcv/operators/OpResize.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> ResizeInto(Tensor &input, Tensor &output, NVCVInterpolationType interp,
                                   std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto resize = CreateOperator<cvop::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*resize});

    resize->submit(pstream->handle(), input.impl(), output.impl(), interp);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Resize(Tensor &input, const Shape &out_shape, NVCVInterpolationType interp,
                               std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(out_shape, input.dtype(), input.layout());

    return ResizeInto(input, *output, interp, pstream);
}

std::shared_ptr<ImageBatchVarShape> ResizeVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                       NVCVInterpolationType interp, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto resize = CreateOperator<cvop::Resize>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*resize});

    resize->submit(pstream->handle(), input.impl(), output.impl(), interp);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> ResizeVarShape(ImageBatchVarShape &input, const std::vector<Size2D> &out_size,
                                                   NVCVInterpolationType interp, std::shared_ptr<Stream> pstream)
{
    if (input.numImages() != (int)out_size.size())
    {
        throw std::runtime_error("Number of input images must be equal to the number of elements in output size list ");
    }

    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        auto            image  = Image::Create(out_size[i], format);
        output->pushBack(*image);
    }

    return ResizeVarShapeInto(input, *output, interp, pstream);
}

} // namespace

void ExportOpResize(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("resize", &Resize, "shape"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
                           "stream"_a = nullptr);
    DefClassMethod<Tensor>("resize_into", &ResizeInto, "out"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
                           "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("resize", &ResizeVarShape, "sizes"_a, "interp"_a = NVCV_INTERP_LINEAR,
                                       py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("resize_into", &ResizeVarShapeInto, "out"_a, "interp"_a = NVCV_INTERP_LINEAR,
                                       py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
