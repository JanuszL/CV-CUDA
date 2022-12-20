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
#include <nvcv/operators/OpMedianBlur.hpp>
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
Tensor MedianBlurInto(Tensor &input, Tensor &output, const std::tuple<int, int> &ksize, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto median_blur = CreateOperator<cvop::MedianBlur>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*median_blur});

    cv::Size2D ksizeArg{std::get<0>(ksize), std::get<1>(ksize)};

    median_blur->submit(pstream->cudaHandle(), input, output, ksizeArg);

    return output;
}

Tensor MedianBlur(Tensor &input, const std::tuple<int, int> &ksize, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return MedianBlurInto(input, output, ksize, pstream);
}

ImageBatchVarShape VarShapeMedianBlurInto(ImageBatchVarShape &input, ImageBatchVarShape &output, Tensor &ksize,
                                          std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto median_blur = CreateOperator<cvop::MedianBlur>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, ksize});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*median_blur});

    median_blur->submit(pstream->cudaHandle(), input, output, ksize);

    return output;
}

ImageBatchVarShape VarShapeMedianBlur(ImageBatchVarShape &input, Tensor &ksize, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input[i].format();
        cv::Size2D      size   = input[i].size();
        auto            image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return VarShapeMedianBlurInto(input, output, ksize, pstream);
}

} // namespace

void ExportOpMedianBlur(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::Tensor>("median_blur", &MedianBlur, "ksize"_a, py::kw_only(), "stream"_a = nullptr);
    util::DefClassMethod<priv::Tensor>("median_blur_into", &MedianBlurInto, "output"_a, "ksize"_a, py::kw_only(),
                                       "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("median_blur", &VarShapeMedianBlur, "ksize"_a, py::kw_only(),
                                                   "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>("median_blur_into", &VarShapeMedianBlurInto, "output"_a, "ksize"_a,
                                                   py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
