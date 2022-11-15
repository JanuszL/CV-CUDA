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
#include <nvcv/operators/OpChannelReorder.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <nvcv/python/Image.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {

ImageBatchVarShape ChannelReorderVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input, Tensor &orders,
                                              std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto chReorder = CreateOperator<cvop::ChannelReorder>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, orders});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*chReorder});

    chReorder->submit(pstream->cudaHandle(), input, output, orders);

    return output;
}

ImageBatchVarShape ChannelReorderVarShape(ImageBatchVarShape &input, Tensor &orders, std::optional<cv::ImageFormat> fmt,
                                          std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = fmt ? *fmt : input[i].format();
        cv::Size2D      size   = input[i].size();
        auto            image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return ChannelReorderVarShapeInto(output, input, orders, pstream);
}

} // namespace

void ExportOpChannelReorder(py::module &m)
{
    using namespace pybind11::literals;

    m.def("channelreorder", &ChannelReorderVarShape, "src"_a, "order"_a, py::kw_only(), "format"_a = nullptr,
          "stream"_a = nullptr);
    m.def("channelreorder_into", &ChannelReorderVarShapeInto, "dst"_a, "src"_a, "orders"_a, py::kw_only(),
          "stream"_a = nullptr);
}

} // namespace nv::cvpy
