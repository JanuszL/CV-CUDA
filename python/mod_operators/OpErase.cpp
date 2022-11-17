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
#include <nvcv/operators/OpErase.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
Tensor EraseInto(Tensor &input, Tensor &output, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx,
                 bool random, unsigned int seed, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (anchor.layout().rank() != 1 || anchor.layout()[0] != 'N')
    {
        throw std::runtime_error("Layout of anchor must be 'N'.");
    }

    cv::TensorShape shape = anchor.shape();

    auto erase = CreateOperator<cvop::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*erase});

    erase->submit(pstream->cudaHandle(), input, output, anchor, erasing, values, imgIdx, random, seed);

    return output;
}

Tensor Erase(Tensor &input, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx, bool random,
             unsigned int seed, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return EraseInto(input, output, anchor, erasing, values, imgIdx, random, seed, pstream);
}

ImageBatchVarShape EraseVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output, Tensor &anchor,
                                     Tensor &erasing, Tensor &values, Tensor &imgIdx, bool random, unsigned int seed,
                                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    if (anchor.layout().rank() != 1 || anchor.layout()[0] != 'N')
    {
        throw std::runtime_error("Layout of anchor must be 'N'.");
    }

    cv::TensorShape shape = anchor.shape();

    auto erase = CreateOperator<cvop::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*erase});

    erase->submit(pstream->cudaHandle(), input, output, anchor, erasing, values, imgIdx, random, seed);

    return output;
}

ImageBatchVarShape EraseVarShape(ImageBatchVarShape &input, Tensor &anchor, Tensor &erasing, Tensor &values,
                                 Tensor &imgIdx, bool random, unsigned int seed, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    for (auto img = input.begin(); img != input.end(); ++img)
    {
        auto newimg = Image::Create(img->size(), format);
        output.pushBack(newimg);
    }

    return EraseVarShapeInto(input, output, anchor, erasing, values, imgIdx, random, seed, pstream);
}

} // namespace

void ExportOpErase(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::Tensor>("erase", &Erase, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a, py::kw_only(),
                                       "random"_a = false, "seed"_a = 0, "stream"_a = nullptr);
    util::DefClassMethod<priv::Tensor>("erase_into", &EraseInto, "out"_a, "anchor"_a, "erasing"_a, "values"_a,
                                       "imgIdx"_a, py::kw_only(), "random"_a = false, "seed"_a = 0,
                                       "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("erase", &EraseVarShape, "anchor"_a, "erasing"_a, "values"_a,
                                                   "imgIdx"_a, py::kw_only(), "random"_a = false, "seed"_a = 0,
                                                   "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("erase_into", &EraseVarShapeInto, "out"_a, "anchor"_a, "erasing"_a,
                                                   "values"_a, "imgIdx"_a, py::kw_only(), "random"_a = false,
                                                   "seed"_a = 0, "stream"_a = nullptr);
}

} // namespace nv::cvpy
