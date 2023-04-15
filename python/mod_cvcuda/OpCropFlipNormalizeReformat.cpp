/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpCropFlipNormalizeReformat.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

#include <iostream>

namespace cvcudapy {

namespace {

} // namespace

namespace {
Tensor CropFlipNormalizeReformatInto(Tensor &output, ImageBatchVarShape &input, Tensor &cropRect, Tensor &flipCode,
                                     Tensor &base, Tensor &scale, float globalScale, float globalShift, float epsilon,
                                     std::optional<uint32_t> flags, NVCVBorderType borderMode, float borderValue,
                                     std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto op = CreateOperator<cvcuda::CropFlipNormalizeReformat>();

    if (!flags)
    {
        flags = 0;
    }

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, cropRect, flipCode, base, scale});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_NONE, {*op});
    op->submit(pstream->cudaHandle(), input, output, cropRect, borderMode, borderValue, flipCode, base, scale,
               globalScale, globalShift, epsilon, *flags);

    return output;
}

Tensor CropFlipNormalizeReformat(ImageBatchVarShape &input, const Shape &out_shape, nvcv::DataType out_dtype,
                                 nvcv::TensorLayout out_layout, Tensor &cropRect, Tensor &flipCode, Tensor &base,
                                 Tensor &scale, float globalScale, float globalShift, float epsilon,
                                 std::optional<uint32_t> flags, NVCVBorderType borderMode, float borderValue,
                                 std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(out_shape, out_dtype, out_layout);

    return CropFlipNormalizeReformatInto(output, input, cropRect, flipCode, base, scale, globalScale, globalShift,
                                         epsilon, flags, borderMode, borderValue, pstream);
}

} // namespace

void ExportOpCropFlipNormalizeReformat(py::module &m)
{
    using namespace pybind11::literals;

    float defGlobalScale = 1;
    float defGlobalShift = 0;
    float defEpsilon     = 0;

    m.def("crop_flip_normalize_reformat", &CropFlipNormalizeReformat, "src"_a, "out_shape"_a, "out_dtype"_a,
          "out_layout"_a, "rect"_a, "flip_code"_a, "base"_a, "scale"_a, "globalscale"_a = defGlobalScale,
          "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon, "flags"_a = std::nullopt,
          "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(), "stream"_a = nullptr);
    m.def("crop_flip_normalize_reformat_into", &CropFlipNormalizeReformatInto, "dst"_a, "src"_a, "rect"_a,
          "flip_code"_a, "base"_a, "scale"_a, "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
          "epsilon"_a = defEpsilon, "flags"_a = std::nullopt, "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0,
          py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy
