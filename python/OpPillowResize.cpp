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

#include "ImageFormat.hpp"
#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <nvcv/TensorDataAccess.hpp>
#include <operators/OpPillowResize.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> PillowResizeInto(Tensor &input, Tensor &output, cv::ImageFormat format,
                                         NVCVInterpolationType interp, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }
    auto in_access  = cv::TensorDataAccessPitchImagePlanar::Create(*input.impl().exportData());
    auto out_access = cv::TensorDataAccessPitchImagePlanar::Create(*output.impl().exportData());
    if (!in_access || !out_access)
    {
        throw std::runtime_error("Incompatible input/output tensor layout");
    }
    int        w              = std::max(in_access->numCols(), out_access->numCols());
    int        h              = std::max(in_access->numRows(), out_access->numRows());
    int        max_batch_size = in_access->numSamples();
    cv::Size2D size{w, h};
    auto       PillowResize = CreateOperator<cvop::PillowResize>(size, max_batch_size, format);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*PillowResize});

    PillowResize->submit(pstream->handle(), input.impl(), output.impl(), interp);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> PillowResize(Tensor &input, const Shape &out_shape, cv::ImageFormat format,
                                     NVCVInterpolationType interp, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(out_shape, input.dtype(), input.layout());

    return PillowResizeInto(input, *output, format, interp, pstream);
}

} // namespace

void ExportOpPillowResize(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("pillowresize", &PillowResize, "shape"_a, "format"_a, "interp"_a = NVCV_INTERP_LINEAR,
                           py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("pillowresize_into", &PillowResizeInto, "out"_a, "format"_a, "interp"_a = NVCV_INTERP_LINEAR,
                           py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
