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

#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <operators/OpResize.hpp>
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

} // namespace

void ExportOpResize(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("resize", &Resize, "shape"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
                           "stream"_a = nullptr);
    DefClassMethod<Tensor>("resize_into", &ResizeInto, "out"_a, "interp"_a = NVCV_INTERP_LINEAR, py::kw_only(),
                           "stream"_a = nullptr);
}

} // namespace nv::cvpy
