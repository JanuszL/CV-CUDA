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
#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <nvcv/TensorLayoutInfo.hpp>
#include <operators/OpCustomCrop.hpp>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> CustomCropInto(Tensor &input, Tensor &output, const NVCVRectI &rcCrop,
                                       std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto crop = CreateOperator<cvop::CustomCrop>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*crop});

    crop->submit(pstream->handle(), input.impl(), output.impl(), rcCrop);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> CustomCrop(Tensor &input, const NVCVRectI &rcCrop, std::shared_ptr<Stream> pstream)
{
    auto info = cv::TensorLayoutInfoImage::Create(input.layout() ? *input.layout() : cv::TensorLayout::NONE);
    if (!info)
    {
        throw std::invalid_argument("Non-supported tensor layout");
    }

    int iwidth  = info->idxWidth();
    int iheight = info->idxHeight();

    NVCV_ASSERT(iwidth >= 0 && "All images have width");

    // If no height, we consider height==1, and this dimension can't be changed
    // in order to keep the output layout the same as input's
    if (iheight < 0 && rcCrop.height != 1)
    {
        throw std::invalid_argument("Non-supported tensor layout");
    }

    // Create the output shape based inputs, changing width/height to match rcCrop's size
    Shape out_shape   = input.shape();
    out_shape[iwidth] = rcCrop.width;
    if (iheight >= 0)
    {
        out_shape[iheight] = rcCrop.height;
    }

    std::shared_ptr<Tensor> output = Tensor::Create(out_shape, input.dtype(), input.layout());

    return CustomCropInto(input, *output, rcCrop, pstream);
}

} // namespace

void ExportOpCustomCrop(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("customcrop", &CustomCrop, "rect"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("customcrop_into", &CustomCropInto, "out"_a, "rect"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
