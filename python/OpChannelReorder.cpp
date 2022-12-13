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

#include "Image.hpp"
#include "ImageBatch.hpp"
#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "String.hpp"
#include "Tensor.hpp"
#include "operators/Types.h"

#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpChannelReorder.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {

std::shared_ptr<ImageBatchVarShape> ChannelReorderVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                               Tensor &orders, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto chReorder = CreateOperator<cvop::ChannelReorder>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, orders});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*chReorder});

    chReorder->submit(pstream->handle(), input.impl(), output.impl(), orders.impl());

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> ChannelReorderVarShape(ImageBatchVarShape &input, Tensor &orders,
                                                           std::optional<cv::ImageFormat> fmt,
                                                           std::shared_ptr<Stream>        pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = fmt ? *fmt : input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return ChannelReorderVarShapeInto(input, *output, orders, pstream);
}

} // namespace

void ExportOpChannelReorder(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<ImageBatchVarShape>("channelreorder", &ChannelReorderVarShape, "order"_a, py::kw_only(),
                                       "format"_a = nullptr, "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("channelreorder_into", &ChannelReorderVarShapeInto, "output"_a, "orders"_a,
                                       py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
