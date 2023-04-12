/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpBndBox.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {
Tensor BndBoxInto(Tensor &output, Tensor &input,
                  const NVCVRectI &bbox, int thickness, uchar4 borderColor, uchar4 fillColor, bool enableMSAA,
                  std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::BndBox>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*op});

    op->submit(pstream->cudaHandle(), input, output, bbox, thickness, borderColor, fillColor, enableMSAA);

    return std::move(output);
}

Tensor BndBox(Tensor &input, const NVCVRectI &bbox, int thickness, uchar4 borderColor, uchar4 fillColor, bool enableMSAA, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return BndBoxInto(output, input, bbox, thickness, borderColor, fillColor, enableMSAA, pstream);
}

} // namespace

void ExportOpBndBox(py::module &m)
{
    using namespace pybind11::literals;

    m.def("bndbox", &BndBox, "src"_a, "bbox"_a, "thickness"_a, "borderColor"_a, "fillColor"_a, "enableMSAA"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("bndbox_into", &BndBoxInto, "dst"_a, "src"_a, "bbox"_a, "thickness"_a, "borderColor"_a, "fillColor"_a, "enableMSAA"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy
