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
#include <core/DataType.hpp>
#include <core/ResourceGuard.hpp>
#include <core/Stream.hpp>
#include <core/Tensor.hpp>
#include <nvcv/operators/OpConvertTo.hpp>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> ConvertToInto(Tensor &input, Tensor &output, float scale, float offset,
                                      std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto cvt = CreateOperator<cvop::ConvertTo>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*cvt});

    cvt->submit(pstream->handle(), input.impl(), output.impl(), scale, offset);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> ConvertTo(Tensor &input, cv::DataType dtype, float scale, float offset,
                                  std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), dtype, input.layout());

    return ConvertToInto(input, *output, scale, offset, pstream);
}

} // namespace

void ExportOpConvertTo(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("convertto", &ConvertTo, "dtype"_a, "scale"_a = 1, "offset"_a = 0, py::kw_only(),
                           "stream"_a = nullptr);
    DefClassMethod<Tensor>("convertto_into", &ConvertToInto, "out"_a, "scale"_a = 1, "offset"_a = 0, py::kw_only(),
                           "stream"_a = nullptr);
}

} // namespace nv::cvpy
