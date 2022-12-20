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
#include <nvcv/operators/OpPadAndStack.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace nv::cvpy {

namespace {
Tensor PadAndStackInto(ImageBatchVarShape &input, Tensor &output, Tensor &top, Tensor &left, NVCVBorderType border,
                       float borderValue, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto padstack = CreateOperator<cvop::PadAndStack>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, top, left});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*padstack});

    padstack->submit(pstream->cudaHandle(), input, output, top, left, border, borderValue);

    return std::move(output);
}

Tensor PadAndStack(ImageBatchVarShape &input, Tensor &top, Tensor &left, NVCVBorderType border, float borderValue,
                   std::optional<Stream> pstream)
{
    cv::ImageFormat fmt = input.uniqueFormat();
    if (fmt == cv::FMT_NONE)
    {
        throw std::runtime_error("All images in the input must have the same format");
    }

    Tensor output = Tensor::CreateForImageBatch(input.numImages(), input.maxSize(), fmt);

    return PadAndStackInto(input, output, top, left, border, borderValue, pstream);
}

} // namespace

void ExportOpPadAndStack(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::ImageBatchVarShape>("padandstack", &PadAndStack, "top"_a, "left"_a,
                                                   "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(),
                                                   "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("padandstack_into", &PadAndStackInto, "out"_a, "top"_a, "left"_a,
                                                   "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(),
                                                   "stream"_a = nullptr);
}

} // namespace nv::cvpy
