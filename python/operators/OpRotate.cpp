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
#include <core/Image.hpp>
#include <core/ImageBatch.hpp>
#include <core/ResourceGuard.hpp>
#include <core/Stream.hpp>
#include <core/Tensor.hpp>
#include <nvcv/operators/OpRotate.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> RotateInto(Tensor &input, Tensor &output, double angleDeg,
                                   const std::tuple<double, double> &shift, NVCVInterpolationType interpolation,
                                   std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto rotate = CreateOperator<cvop::Rotate>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*rotate});

    double2 shiftArg{std::get<0>(shift), std::get<1>(shift)};

    rotate->submit(pstream->handle(), input.impl(), output.impl(), angleDeg, shiftArg, interpolation);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Rotate(Tensor &input, double angleDeg, const std::tuple<double, double> &shift,
                               const NVCVInterpolationType interpolation, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return RotateInto(input, *output, angleDeg, shift, interpolation, pstream);
}

std::shared_ptr<ImageBatchVarShape> VarShapeRotateInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                       Tensor &angleDeg, Tensor &shift,
                                                       NVCVInterpolationType   interpolation,
                                                       std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto rotate = CreateOperator<cvop::Rotate>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, angleDeg, shift});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*rotate});

    rotate->submit(pstream->handle(), input.impl(), output.impl(), angleDeg.impl(), shift.impl(), interpolation);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> VarShapeRotate(ImageBatchVarShape &input, Tensor &angleDeg, Tensor &shift,
                                                   NVCVInterpolationType interpolation, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return VarShapeRotateInto(input, *output, angleDeg, shift, interpolation, pstream);
}

} // namespace

void ExportOpRotate(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("rotate", &Rotate, "angle_deg"_a, "shift"_a, "interpolation"_a, py::kw_only(),
                           "stream"_a = nullptr);
    DefClassMethod<Tensor>("rotate_into", &RotateInto, "output"_a, "angle_deg"_a, "shift"_a, "interpolation"_a,
                           py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("rotate", &VarShapeRotate, "angle_deg"_a, "shift"_a, "interpolation"_a,
                                       py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("rotate_into", &VarShapeRotateInto, "output"_a, "angle_deg"_a, "shift"_a,
                                       "interpolation"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
