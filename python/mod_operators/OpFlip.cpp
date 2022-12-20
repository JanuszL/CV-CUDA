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
#include <nvcv/operators/OpFlip.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
Tensor FlipInto(Tensor &input, Tensor &output, int32_t flipCode, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto Flip = CreateOperator<cvop::Flip>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*Flip});

    Flip->submit(pstream->cudaHandle(), input, output, flipCode);

    return output;
}

Tensor Flip(Tensor &input, int32_t flipCode, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return FlipInto(input, output, flipCode, pstream);
}

ImageBatchVarShape FlipVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output, Tensor &flipCode,
                                    std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto flip = CreateOperator<cvop::Flip>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, flipCode});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*flip});

    flip->submit(pstream->cudaHandle(), input, output, flipCode);

    return output;
}

ImageBatchVarShape FlipVarShape(ImageBatchVarShape &input, Tensor &flipCode, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input[i].format();
        cv::Size2D      size   = input[i].size();
        auto            image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return FlipVarShapeInto(input, output, flipCode, pstream);
}

} // namespace

void ExportOpFlip(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::Tensor>("flip", &Flip, "flipCode"_a, py::kw_only(), "stream"_a = nullptr);
    util::DefClassMethod<priv::Tensor>("flip_into", &FlipInto, "output"_a, "flipCode"_a, py::kw_only(),
                                       "stream"_a = nullptr);

    util::DefClassMethod<priv::ImageBatchVarShape>("flip", &FlipVarShape, "flipCode"_a, py::kw_only(),
                                                   "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("flip_into", &FlipVarShapeInto, "output"_a, "flipCode"_a,
                                                   py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
