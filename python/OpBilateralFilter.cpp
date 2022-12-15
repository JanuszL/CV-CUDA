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
#include <operators/OpBilateralFilter.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> BilateralFilterInto(Tensor &input, Tensor &output, int diameter, float sigmaColor,
                                            float sigmaSpace, NVCVBorderType borderMode,
                                            std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto bilateral_filter = CreateOperator<cvop::BilateralFilter>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*bilateral_filter});

    bilateral_filter->submit(pstream->handle(), input.impl(), output.impl(), diameter, sigmaColor, sigmaSpace,
                             borderMode);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> BilateralFilter(Tensor &input, int diameter, float sigmaColor, float sigmaSpace,
                                        NVCVBorderType borderMode, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return BilateralFilterInto(input, *output, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
}

std::shared_ptr<ImageBatchVarShape> VarShapeBilateralFilterInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                                Tensor &diameter, Tensor &sigmaColor,
                                                                Tensor &sigmaSpace, NVCVBorderType borderMode,
                                                                std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto bilateral_filter = CreateOperator<cvop::BilateralFilter>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, diameter, sigmaColor, sigmaSpace});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*bilateral_filter});

    bilateral_filter->submit(pstream->handle(), input.impl(), output.impl(), diameter.impl(), sigmaColor.impl(),
                             sigmaSpace.impl(), borderMode);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> VarShapeBilateralFilter(ImageBatchVarShape &input, Tensor &diameter,
                                                            Tensor &sigmaColor, Tensor &sigmaSpace,
                                                            NVCVBorderType borderMode, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return VarShapeBilateralFilterInto(input, *output, diameter, sigmaColor, sigmaSpace, borderMode, pstream);
}

} // namespace

void ExportOpBilateralFilter(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("bilateral_filter", &BilateralFilter, "diameter"_a, "sigma_color"_a, "sigma_space"_a,
                           py::kw_only(), "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
    DefClassMethod<Tensor>("bilateral_filter_into", &BilateralFilterInto, "output"_a, "diameter"_a, "sigma_color"_a,
                           "sigma_space"_a, py::kw_only(), "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
                           "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("bilateral_filter", &VarShapeBilateralFilter, "diameter"_a, "sigma_color"_a,
                                       "sigma_space"_a, py::kw_only(),
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("bilateral_filter_into", &VarShapeBilateralFilterInto, "output"_a, "diameter"_a,
                                       "sigma_color"_a, "sigma_space"_a, py::kw_only(),
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
}

} // namespace nv::cvpy
