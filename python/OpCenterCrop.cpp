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

#include "Assert.hpp"
#include "Image.hpp"
#include "ImageBatch.hpp"
#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "String.hpp"
#include "Tensor.hpp"
#include "operators/Types.h"

#include <nvcv/TensorLayoutInfo.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpCenterCrop.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> CenterCropInto(Tensor &input, Tensor &output, const std::tuple<int, int> &cropSize,
                                       std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto center_crop = CreateOperator<cvop::CenterCrop>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*center_crop});

    cv::Size2D cropSizeArg{std::get<0>(cropSize), std::get<1>(cropSize)};

    center_crop->submit(pstream->handle(), input.impl(), output.impl(), cropSizeArg);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> CenterCrop(Tensor &input, const std::tuple<int, int> &cropSize, std::shared_ptr<Stream> pstream)
{
    auto info = cv::TensorLayoutInfoImage::Create(input.layout() ? *input.layout() : cv::TensorLayout::NONE);
    if (!info)
    {
        throw std::invalid_argument("Non-supported tensor layout");
    }

    int iwidth  = info->idxWidth();
    int iheight = info->idxHeight();

    NVCV_ASSERT(iwidth >= 0 && "All images have width");
    NVCV_ASSERT(iheight >= 0 && "All images have height");

    // Use cropSize (width, height) for output
    Shape out_shape    = input.shape();
    out_shape[iwidth]  = std::get<0>(cropSize);
    out_shape[iheight] = std::get<1>(cropSize);

    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return CenterCropInto(input, *output, cropSize, pstream);
}

} // namespace

void ExportOpCenterCrop(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("center_crop", &CenterCrop, "crop_size"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("center_crop_into", &CenterCropInto, "output"_a, "crop_size"_a, py::kw_only(),
                           "stream"_a = nullptr);
}

} // namespace nv::cvpy