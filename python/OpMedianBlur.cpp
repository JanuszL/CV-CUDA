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
#include <operators/OpMedianBlur.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> MedianBlurInto(Tensor &input, Tensor &output, const std::tuple<int, int> &ksize,
                                       std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto median_blur = CreateOperator<cvop::MedianBlur>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*median_blur});

    cv::Size2D ksizeArg{std::get<0>(ksize), std::get<1>(ksize)};

    median_blur->submit(pstream->handle(), input.impl(), output.impl(), ksizeArg);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> MedianBlur(Tensor &input, const std::tuple<int, int> &ksize, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return MedianBlurInto(input, *output, ksize, pstream);
}

std::shared_ptr<ImageBatchVarShape> VarShapeMedianBlurInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                           Tensor &ksize, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto median_blur = CreateOperator<cvop::MedianBlur>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, ksize});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*median_blur});

    median_blur->submit(pstream->handle(), input.impl(), output.impl(), ksize.impl());

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> VarShapeMedianBlur(ImageBatchVarShape &input, Tensor &ksize,
                                                       std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return VarShapeMedianBlurInto(input, *output, ksize, pstream);
}

} // namespace

void ExportOpMedianBlur(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("median_blur", &MedianBlur, "ksize"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("median_blur_into", &MedianBlurInto, "output"_a, "ksize"_a, py::kw_only(),
                           "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("median_blur", &VarShapeMedianBlur, "ksize"_a, py::kw_only(),
                                       "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("median_blur_into", &VarShapeMedianBlurInto, "output"_a, "ksize"_a,
                                       py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
