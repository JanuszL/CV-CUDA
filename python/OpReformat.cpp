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
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <operators/OpReformat.hpp>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> ReformatInto(Tensor &input, Tensor &output, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto reformat = CreateOperator<cvop::Reformat>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*reformat});

    reformat->submit(pstream->handle(), input.impl(), output.impl());

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Reformat(Tensor &input, const cv::TensorLayout &out_layout, std::shared_ptr<Stream> pstream)
{
    cv::TensorShape out_shape = Permute(input.impl().shape(), out_layout);

    std::shared_ptr<Tensor> output = Tensor::Create(CreateShape(out_shape), input.dtype(), out_layout);

    return ReformatInto(input, *output, pstream);
}

} // namespace

void ExportOpReformat(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("reformat", &Reformat, "layout"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("reformat_into", &ReformatInto, "out"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
