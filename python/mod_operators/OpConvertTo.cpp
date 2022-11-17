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
#include <nvcv/operators/OpConvertTo.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace nv::cvpy {

namespace {
Tensor ConvertToInto(Tensor &input, Tensor &output, float scale, float offset, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto cvt = CreateOperator<cvop::ConvertTo>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*cvt});

    cvt->submit(pstream->cudaHandle(), input, output, scale, offset);

    return std::move(output);
}

Tensor ConvertTo(Tensor &input, cv::DataType dtype, float scale, float offset, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), dtype);

    return ConvertToInto(input, output, scale, offset, pstream);
}

} // namespace

void ExportOpConvertTo(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::Tensor>("convertto", &ConvertTo, "dtype"_a, "scale"_a = 1, "offset"_a = 0, py::kw_only(),
                                       "stream"_a = nullptr);
    util::DefClassMethod<priv::Tensor>("convertto_into", &ConvertToInto, "out"_a, "scale"_a = 1, "offset"_a = 0,
                                       py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
