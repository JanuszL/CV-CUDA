/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "Container.hpp"
#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "String.hpp"
#include "Tensor.hpp"
#include "operators/Types.h"

#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpRotate.hpp>
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

} // namespace

void ExportOpRotate(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("rotate", &Rotate, "angle_deg"_a, "shift"_a, "interpolation"_a, py::kw_only(),
                           "stream"_a = nullptr);
    DefClassMethod<Tensor>("rotate_into", &RotateInto, "output"_a, "angle_deg"_a, "shift"_a, "interpolation"_a,
                           py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
