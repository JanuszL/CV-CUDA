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
#include <nvcv/operators/OpBilateralFilter.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
Tensor BilateralFilterInto(Tensor &input, Tensor &output, int diameter, float sigmaColor, float sigmaSpace,
                           NVCVBorderType borderMode, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto bilateral_filter = CreateOperator<cvop::BilateralFilter>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*bilateral_filter});

    bilateral_filter->submit(pstream->cudaHandle(), input, output, diameter, sigmaColor, sigmaSpace, borderMode);

    return output;
}

Tensor BilateralFilter(Tensor &input, int diameter, float sigmaColor, float sigmaSpace, NVCVBorderType borderMode,
                       std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return BilateralFilterInto(input, output, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
}

ImageBatchVarShape VarShapeBilateralFilterInto(ImageBatchVarShape &input, ImageBatchVarShape &output, Tensor &diameter,
                                               Tensor &sigmaColor, Tensor &sigmaSpace, NVCVBorderType borderMode,
                                               std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto bilateral_filter = CreateOperator<cvop::BilateralFilter>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, diameter, sigmaColor, sigmaSpace});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*bilateral_filter});

    bilateral_filter->submit(pstream->cudaHandle(), input, output, diameter, sigmaColor, sigmaSpace, borderMode);

    return output;
}

ImageBatchVarShape VarShapeBilateralFilter(ImageBatchVarShape &input, Tensor &diameter, Tensor &sigmaColor,
                                           Tensor &sigmaSpace, NVCVBorderType borderMode, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input[i].format();
        cv::Size2D      size   = input[i].size();
        auto            image  = Image::Create(size, format);
        output.pushBack(image);
    }

    return VarShapeBilateralFilterInto(input, output, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
}

} // namespace

void ExportOpBilateralFilter(py::module &m)
{
    using namespace pybind11::literals;

    util::DefClassMethod<priv::Tensor>("bilateral_filter", &BilateralFilter, "diameter"_a, "sigma_color"_a,
                                       "sigma_space"_a, py::kw_only(),
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
    util::DefClassMethod<priv::Tensor>("bilateral_filter_into", &BilateralFilterInto, "output"_a, "diameter"_a,
                                       "sigma_color"_a, "sigma_space"_a, py::kw_only(),
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);

    util::DefClassMethod<priv::ImageBatchVarShape>(
        "bilateral_filter", &VarShapeBilateralFilter, "diameter"_a, "sigma_color"_a, "sigma_space"_a, py::kw_only(),
        "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
    util::DefClassMethod<priv::ImageBatchVarShape>(
        "bilateral_filter_into", &VarShapeBilateralFilterInto, "output"_a, "diameter"_a, "sigma_color"_a,
        "sigma_space"_a, py::kw_only(), "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
}

} // namespace nv::cvpy
