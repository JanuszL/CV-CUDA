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

#include <common/Assert.hpp>
#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <nvcv/TensorLayoutInfo.hpp>
#include <nvcv/operators/OpCenterCrop.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
Tensor CenterCropInto(Tensor &input, Tensor &output, const std::tuple<int, int> &cropSize,
                      std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto center_crop = CreateOperator<cvop::CenterCrop>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*center_crop});

    cv::Size2D cropSizeArg{std::get<0>(cropSize), std::get<1>(cropSize)};

    center_crop->submit(pstream->cudaHandle(), input, output, cropSizeArg);

    return output;
}

Tensor CenterCrop(Tensor &input, const std::tuple<int, int> &cropSize, std::optional<Stream> pstream)
{
    auto info = cv::TensorLayoutInfoImage::Create(input.layout());
    if (!info)
    {
        throw std::invalid_argument("Non-supported tensor layout");
    }

    int iwidth  = info->idxWidth();
    int iheight = info->idxHeight();

    NVCV_ASSERT(iwidth >= 0 && "All images have width");
    NVCV_ASSERT(iheight >= 0 && "All images have height");

    // Use cropSize (width, height) for output
    cv::TensorShape tshape = input.shape();
    Shape           out_shape{&tshape[0], &tshape[0] + tshape.rank()};
    out_shape[iwidth]  = std::get<0>(cropSize);
    out_shape[iheight] = std::get<1>(cropSize);

    Tensor output = Tensor::Create(cv::TensorShape(out_shape.data(), out_shape.size(), input.layout()), input.dtype());

    return CenterCropInto(input, output, cropSize, pstream);
}

} // namespace

void ExportOpCenterCrop(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::Tensor>("center_crop", &CenterCrop, "crop_size"_a, py::kw_only(), "stream"_a = nullptr);
    util::DefClassMethod<priv::Tensor>("center_crop_into", &CenterCropInto, "output"_a, "crop_size"_a, py::kw_only(),
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
