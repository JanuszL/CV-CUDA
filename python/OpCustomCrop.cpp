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

#include "Assert.hpp"
#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <nvcv/TensorLayoutInfo.hpp>
#include <nvcv/operators/OpCustomCrop.hpp>

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
