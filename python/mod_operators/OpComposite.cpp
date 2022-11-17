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
#include <nvcv/operators/OpComposite.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> CompositeInto(Tensor &foreground, Tensor &output, Tensor &background, Tensor &fgMask,
                                      std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto composite = CreateOperator<cvop::Composite>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {foreground, background, fgMask});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*composite});

    composite->submit(pstream->handle(), foreground.impl(), background.impl(), fgMask.impl(), output.impl());

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Composite(Tensor &foreground, Tensor &background, Tensor &fgMask, int outChannels,
                                  std::shared_ptr<Stream> pstream)
{
    Shape fg_shape = foreground.shape();
    Shape out_shape(fg_shape);
    int   cdim          = out_shape.size();
    out_shape[cdim - 1] = outChannels;

    std::shared_ptr<Tensor> output = Tensor::Create(out_shape, foreground.dtype(), foreground.layout());

    return CompositeInto(foreground, *output, background, fgMask, pstream);
}

std::shared_ptr<ImageBatchVarShape> CompositeVarShapeInto(ImageBatchVarShape &foreground, ImageBatchVarShape &output,
                                                          ImageBatchVarShape &background, ImageBatchVarShape &fgMask,
                                                          std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto composite = CreateOperator<cvop::Composite>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {foreground, background, fgMask});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*composite});

    composite->submit(pstream->handle(), foreground.impl(), background.impl(), fgMask.impl(), output.impl());

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> CompositeVarShape(ImageBatchVarShape &foreground, ImageBatchVarShape &background,
                                                      ImageBatchVarShape &fgMask, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(foreground.numImages());

    cv::ImageFormat format = foreground.impl().uniqueFormat();

    for (auto img = foreground.impl().begin(); img != foreground.impl().end(); ++img)
    {
        Size2D size   = {img->size().w, img->size().h};
        auto   newimg = Image::Create(size, format);
        output->pushBack(*newimg);
    }

    return CompositeVarShapeInto(foreground, *output, background, fgMask, pstream);
}

} // namespace

void ExportOpComposite(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<Tensor>("composite", &Composite, "background"_a, "fgmask"_a, "outchannels"_a, py::kw_only(),
                                 "stream"_a = nullptr);
    util::DefClassMethod<Tensor>("composite_into", &CompositeInto, "output"_a, "background"_a, "fgmask"_a,
                                 py::kw_only(), "stream"_a = nullptr);
    util::DefClassMethod<ImageBatchVarShape>("composite", &CompositeVarShape, "background"_a, "fgmask"_a, py::kw_only(),
                                             "stream"_a = nullptr);
    util::DefClassMethod<ImageBatchVarShape>("composite_into", &CompositeVarShapeInto, "output"_a, "background"_a,
                                             "fgmask"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
