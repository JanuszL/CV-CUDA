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
#include <nvcv/operators/OpErase.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> EraseInto(Tensor &input, Tensor &output, Tensor &anchor, Tensor &erasing, Tensor &values,
                                  Tensor &imgIdx, bool random, unsigned int seed, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    if (anchor.layout()->rank() != 1 || (*anchor.layout())[0] != 'N')
    {
        throw std::runtime_error("Layout of anchor must be 'N'.");
    }

    Shape shape = anchor.shape();

    auto erase = CreateOperator<cvop::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*erase});

    erase->submit(pstream->handle(), input.impl(), output.impl(), anchor.impl(), erasing.impl(), values.impl(),
                  imgIdx.impl(), random, seed);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Erase(Tensor &input, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx,
                              bool random, unsigned int seed, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return EraseInto(input, *output, anchor, erasing, values, imgIdx, random, seed, pstream);
}

std::shared_ptr<ImageBatchVarShape> EraseVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                      Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx,
                                                      bool random, unsigned int seed, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    if (anchor.layout()->rank() != 1 || (*anchor.layout())[0] != 'N')
    {
        throw std::runtime_error("Layout of anchor must be 'N'.");
    }

    Shape shape = anchor.shape();

    auto erase = CreateOperator<cvop::Erase>((int)shape[0]);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*erase});

    erase->submit(pstream->handle(), input.impl(), output.impl(), anchor.impl(), erasing.impl(), values.impl(),
                  imgIdx.impl(), random, seed);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> EraseVarShape(ImageBatchVarShape &input, Tensor &anchor, Tensor &erasing,
                                                  Tensor &values, Tensor &imgIdx, bool random, unsigned int seed,
                                                  std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.impl().uniqueFormat();
    if (!format)
    {
        throw std::runtime_error("All images in input must have the same format.");
    }

    for (auto img = input.impl().begin(); img != input.impl().end(); ++img)
    {
        Size2D size   = {img->size().w, img->size().h};
        auto   newimg = Image::Create(size, format);
        output->pushBack(*newimg);
    }

    return EraseVarShapeInto(input, *output, anchor, erasing, values, imgIdx, random, seed, pstream);
}

} // namespace

void ExportOpErase(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("erase", &Erase, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a, py::kw_only(),
                           "random"_a = false, "seed"_a = 0, "stream"_a = nullptr);
    DefClassMethod<Tensor>("erase_into", &EraseInto, "out"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a,
                           py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("erase", &EraseVarShape, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a,
                                       py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("erase_into", &EraseVarShapeInto, "out"_a, "anchor"_a, "erasing"_a, "values"_a,
                                       "imgIdx"_a, py::kw_only(), "random"_a = false, "seed"_a = 0,
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
