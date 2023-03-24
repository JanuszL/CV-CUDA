/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cvcuda/OpAdaptiveThreshold.hpp>
#include <nvcv/python/ImageBatchVarShape.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>
#include <pybind11/stl.h>

namespace cvcudapy {

namespace {
Tensor AdaptiveThresholdInto(Tensor &output, Tensor &input, double max_value, NVCVAdaptiveThresholdType adaptive_method,
                             NVCVThresholdType threshold_type, int32_t block_size, double c,
                             std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto adaptiveThreshold = CreateOperator<cvcuda::AdaptiveThreshold>(block_size, 0);

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*adaptiveThreshold});

    adaptiveThreshold->submit(pstream->cudaHandle(), input, output, max_value, adaptive_method, threshold_type,
                              block_size, c);

    return output;
}

Tensor AdaptiveThreshold(Tensor &input, double max_value, NVCVAdaptiveThresholdType adaptive_method,
                         NVCVThresholdType threshold_type, int32_t block_size, double c, std::optional<Stream> pstream)
{
    Tensor output = Tensor::Create(input.shape(), input.dtype());

    return AdaptiveThresholdInto(output, input, max_value, adaptive_method, threshold_type, block_size, c, pstream);
}

ImageBatchVarShape AdaptiveThresholdVarShapeInto(ImageBatchVarShape &output, ImageBatchVarShape &input,
                                                 Tensor &max_value, NVCVAdaptiveThresholdType adaptive_method,
                                                 NVCVThresholdType threshold_type, int32_t max_block_size,
                                                 Tensor &block_size, Tensor &c, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    auto adaptiveThreshold = CreateOperator<cvcuda::AdaptiveThreshold>(max_block_size, input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {input, max_value, block_size, c});
    guard.add(LockMode::LOCK_WRITE, {output});
    guard.add(LockMode::LOCK_WRITE, {*adaptiveThreshold});

    adaptiveThreshold->submit(pstream->cudaHandle(), input, output, max_value, adaptive_method, threshold_type,
                              block_size, c);

    return output;
}

ImageBatchVarShape AdaptiveThresholdVarShape(ImageBatchVarShape &input, Tensor &max_value,
                                             NVCVAdaptiveThresholdType adaptive_method,
                                             NVCVThresholdType threshold_type, int32_t max_block_size,
                                             Tensor &block_size, Tensor &c, std::optional<Stream> pstream)
{
    ImageBatchVarShape output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        output.pushBack(Image::Create(input[i].size(), input[i].format()));
    }

    return AdaptiveThresholdVarShapeInto(output, input, max_value, adaptive_method, threshold_type, max_block_size,
                                         block_size, c, pstream);
}

} // namespace

void ExportOpAdaptiveThreshold(py::module &m)
{
    using namespace pybind11::literals;

    m.def("adaptivethreshold", &AdaptiveThreshold, "src"_a, "max_value"_a,
          "adaptive_method"_a = NVCV_ADAPTIVE_THRESH_MEAN_C, "threshold_type"_a = NVCV_THRESH_BINARY, "block_size"_a,
          "c"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("adaptivethreshold_into", &AdaptiveThresholdInto, "dst"_a, "src"_a, "max_value"_a,
          "adaptive_method"_a = NVCV_ADAPTIVE_THRESH_MEAN_C, "threshold_type"_a = NVCV_THRESH_BINARY, "block_size"_a,
          "c"_a, py::kw_only(), "stream"_a = nullptr);

    m.def("adaptivethreshold", &AdaptiveThresholdVarShape, "src"_a, "max_value"_a,
          "adaptive_method"_a = NVCV_ADAPTIVE_THRESH_MEAN_C, "threshold_type"_a = NVCV_THRESH_BINARY,
          "max_block_size"_a, "block_size"_a, "c"_a, py::kw_only(), "stream"_a = nullptr);
    m.def("adaptivethreshold_into", &AdaptiveThresholdVarShapeInto, "dst"_a, "src"_a, "max_value"_a,
          "adaptive_method"_a = NVCV_ADAPTIVE_THRESH_MEAN_C, "threshold_type"_a = NVCV_THRESH_BINARY,
          "max_block_size"_a, "block_size"_a, "c"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy
