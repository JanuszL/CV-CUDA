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
#include <nvcv/operators/OpReformat.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace nv::cvpy {

namespace {
Tensor ReformatInto(Tensor &output, Tensor &input, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto reformat = CreateOperator<cvop::Reformat>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*reformat});

    reformat->submit(pstream->cudaHandle(), input, output);

    return std::move(output);
}

Tensor Reformat(Tensor &input, const cv::TensorLayout &out_layout, std::optional<Stream> pstream)
{
    cv::TensorShape out_shape = Permute(input.shape(), out_layout);

    Tensor output = Tensor::Create(out_shape, input.dtype());

    return ReformatInto(output, input, pstream);
}

} // namespace

void ExportOpReformat(py::module &m)
{
    using namespace pybind11::literals;

    m.def("reformat", &Reformat, "src"_a, "layout"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("reformat_into", &ReformatInto, "dst"_a, "src"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
