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
#include <nvcv/operators/OpNormalize.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {

enum OpFlags : uint32_t
{
    SCALE_IS_STDDEV = NVCV_OP_NORMALIZE_SCALE_IS_STDDEV
};

} // namespace

namespace {
std::shared_ptr<Tensor> NormalizeInto(Tensor &input, Tensor &output, Tensor &base, Tensor &scale,
                                      std::optional<uint32_t> flags, float globalScale, float globalShift,
                                      float epsilon, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    if (!flags)
    {
        flags = 0;
    }

    auto normalize = CreateOperator<cvop::Normalize>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, base, scale});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*normalize});

    normalize->submit(pstream->handle(), input.impl(), base.impl(), scale.impl(), output.impl(), globalScale,
                      globalShift, epsilon, *flags);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Normalize(Tensor &input, Tensor &base, Tensor &scale, std::optional<uint32_t> flags,
                                  float globalScale, float globalShift, float epsilon, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return NormalizeInto(input, *output, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

std::shared_ptr<ImageBatchVarShape> VarShapeNormalizeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                          Tensor &base, Tensor &scale, std::optional<uint32_t> flags,
                                                          float globalScale, float globalShift, float epsilon,
                                                          std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    if (!flags)
    {
        flags = 0;
    }

    auto normalize = CreateOperator<cvop::Normalize>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, base, scale});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*normalize});

    normalize->submit(pstream->handle(), input.impl(), base.impl(), scale.impl(), output.impl(), globalScale,
                      globalShift, epsilon, *flags);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> VarShapeNormalize(ImageBatchVarShape &input, Tensor &base, Tensor &scale,
                                                      std::optional<uint32_t> flags, float globalScale,
                                                      float globalShift, float epsilon, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return VarShapeNormalizeInto(input, *output, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

} // namespace

void ExportOpNormalize(py::module &m)
{
    using namespace pybind11::literals;

    py::enum_<OpFlags>(m, "NormalizeFlags").value("SCALE_IS_STDDEV", OpFlags::SCALE_IS_STDDEV);

    float defGlobalScale = 1;
    float defGlobalShift = 0;
    float defEpsilon     = 0;

    DefClassMethod<Tensor>("normalize", &Normalize, "base"_a, "scale"_a, "flags"_a = std::nullopt, py::kw_only(),
                           "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
                           "stream"_a = nullptr);

    DefClassMethod<Tensor>("normalize_into", &NormalizeInto, "out"_a, "base"_a, "scale"_a, "flags"_a = std::nullopt,
                           py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
                           "epsilon"_a = defEpsilon, "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("normalize", &VarShapeNormalize, "base"_a, "scale"_a, "flags"_a = std::nullopt,
                                       py::kw_only(), "globalscale"_a = defGlobalScale,
                                       "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
                                       "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("normalize_into", &VarShapeNormalizeInto, "out"_a, "base"_a, "scale"_a,
                                       "flags"_a = std::nullopt, py::kw_only(), "globalscale"_a = defGlobalScale,
                                       "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
