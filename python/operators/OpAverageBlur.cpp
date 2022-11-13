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
#include <core/Image.hpp>
#include <core/ImageBatch.hpp>
#include <core/ResourceGuard.hpp>
#include <core/Stream.hpp>
#include <core/Tensor.hpp>
#include <nvcv/operators/OpAverageBlur.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> AverageBlurInto(Tensor &input, Tensor &output, const std::tuple<int, int> &kernel_size,
                                        const std::tuple<int, int> &kernel_anchor, NVCVBorderType border,
                                        std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    cv::Size2D kernelSizeArg{std::get<0>(kernel_size), std::get<1>(kernel_size)};
    int2       kernelAnchorArg{std::get<0>(kernel_anchor), std::get<1>(kernel_anchor)};

    auto averageBlur = CreateOperator<cvop::AverageBlur>(kernelSizeArg, 0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*averageBlur});

    averageBlur->submit(pstream->handle(), input.impl(), output.impl(), kernelSizeArg, kernelAnchorArg, border);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> AverageBlur(Tensor &input, const std::tuple<int, int> &kernel_size,
                                    const std::tuple<int, int> &kernel_anchor, NVCVBorderType border,
                                    std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return AverageBlurInto(input, *output, kernel_size, kernel_anchor, border, pstream);
}

std::shared_ptr<ImageBatchVarShape> AverageBlurVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                            const std::tuple<int, int> &max_kernel_size,
                                                            Tensor &kernel_size, Tensor &kernel_anchor,
                                                            NVCVBorderType border, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    cv::Size2D maxKernelSizeArg{std::get<0>(max_kernel_size), std::get<1>(max_kernel_size)};

    auto averageBlur = CreateOperator<cvop::AverageBlur>(maxKernelSizeArg, input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, kernel_size, kernel_anchor});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*averageBlur});

    averageBlur->submit(pstream->handle(), input.impl(), output.impl(), kernel_size.impl(), kernel_anchor.impl(),
                        border);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> AverageBlurVarShape(ImageBatchVarShape         &input,
                                                        const std::tuple<int, int> &max_kernel_size,
                                                        Tensor &kernel_size, Tensor &kernel_anchor,
                                                        NVCVBorderType border, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return AverageBlurVarShapeInto(input, *output, max_kernel_size, kernel_size, kernel_anchor, border, pstream);
}

} // namespace

void ExportOpAverageBlur(py::module &m)
{
    using namespace pybind11::literals;

    const std::tuple<int, int> def_anchor{-1, -1};

    DefClassMethod<Tensor>("averageblur", &AverageBlur, "kernel_size"_a, "kernel_anchor"_a = def_anchor,
                           "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("averageblur_into", &AverageBlurInto, "output"_a, "kernel_size"_a,
                           "kernel_anchor"_a = def_anchor, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
                           py::kw_only(), "stream"_a                  = nullptr);

    DefClassMethod<ImageBatchVarShape>("averageblur", &AverageBlurVarShape, "max_kernel_size"_a, "kernel_size"_a,
                                       "kernel_anchor"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
                                       py::kw_only(), "stream"_a     = nullptr);
    DefClassMethod<ImageBatchVarShape>(
        "averageblur_into", &AverageBlurVarShapeInto, "output"_a, "max_kernel_size"_a, "kernel_size"_a,
        "kernel_anchor"_a, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
