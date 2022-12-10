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

    cvop::MedianBlur median_blur(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});

    cv::Size2D ksizeArg{std::get<0>(ksize), std::get<1>(ksize)};

    median_blur(pstream->handle(), input.impl(), output.impl(), ksizeArg);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> MedianBlur(Tensor &input, const std::tuple<int, int> &ksize, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return MedianBlurInto(input, *output, ksize, pstream);
}

} // namespace

void ExportOpMedianBlur(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("median_blur", &MedianBlur, "ksize"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("median_blur_into", &MedianBlurInto, "output"_a, "ksize"_a, py::kw_only(),
                           "stream"_a = nullptr);
}

} // namespace nv::cvpy
