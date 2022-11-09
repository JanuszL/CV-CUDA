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

#include "Image.hpp"
#include "ImageBatch.hpp"
#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "String.hpp"
#include "Tensor.hpp"

#include <nvcv/operators/OpFlip.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> FlipInto(Tensor &input, Tensor &output, int32_t flipCode, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto Flip = CreateOperator<cvop::Flip>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*Flip});

    Flip->submit(pstream->handle(), input.impl(), output.impl(), flipCode);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Flip(Tensor &input, int32_t flipCode, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return FlipInto(input, *output, flipCode, pstream);
}

std::shared_ptr<ImageBatchVarShape> FlipVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                     Tensor &flipCode, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto Flip = CreateOperator<cvop::Flip>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, flipCode});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*Flip});

    Flip->submit(pstream->handle(), input.impl(), output.impl(), flipCode.impl());

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> FlipVarShape(ImageBatchVarShape &input, Tensor &flipCode,
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

    return FlipVarShapeInto(input, *output, flipCode, pstream);
}

} // namespace

void ExportOpFlip(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("flip", &Flip, "flipCode"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("flip_into", &FlipInto, "output"_a, "flipCode"_a, py::kw_only(), "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("flip", &FlipVarShape, "flipCode"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("flip_into", &FlipVarShapeInto, "output"_a, "flipCode"_a, py::kw_only(),
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
