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
#include <mod_core/Image.hpp>
#include <mod_core/ImageBatch.hpp>
#include <mod_core/ResourceGuard.hpp>
#include <mod_core/Stream.hpp>
#include <mod_core/Tensor.hpp>
#include <nvcv/operators/OpWarpPerspective.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {

using pyarray = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::shared_ptr<Tensor> WarpPerspectiveInto(Tensor &input, Tensor &output, const pyarray &xform, const int32_t flags,
                                            const NVCVBorderType borderMode, const pyarray &borderValue,
                                            std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    size_t bValueSize = borderValue.size();
    size_t bValueDims = borderValue.ndim();
    if (bValueSize > 4 || bValueDims != 1)
    {
        throw std::runtime_error(FormatString(
            "Channels of borderValue should <= 4 and dimension should be 2, current is '%lu', '%lu' respectively",
            bValueSize, bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nv::cv::cuda::GetElement(bValue, i) = bValueSize > i ? *borderValue.data(i) : 0.f;
    }

    size_t xformDims = xform.ndim();
    if (!(xformDims == 2 && xform.shape(0) == 3 && xform.shape(1) == 3))
    {
        throw std::runtime_error(
            FormatString("Details of transformation matrix: nDim == 2, shape == (3, 3) but current is "
                         "'%lu', ('%lu', '%lu') respectively",
                         xformDims, xform.shape(0), xform.shape(1)));
    }

    NVCVPerspectiveTransform xformOutput;
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            xformOutput[i * 3 + j] = *xform.data(i, j);
        }
    }

    auto warpPerspective = CreateOperator<cvop::WarpPerspective>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*warpPerspective});

    warpPerspective->submit(pstream->handle(), input.impl(), output.impl(), xformOutput, flags, borderMode, bValue);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> WarpPerspective(Tensor &input, const pyarray &xform, const int32_t flags,
                                        const NVCVBorderType borderMode, const pyarray &borderValue,
                                        std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return WarpPerspectiveInto(input, *output, xform, flags, borderMode, borderValue, pstream);
}

std::shared_ptr<ImageBatchVarShape> WarpPerspectiveVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                                Tensor &xform, const int32_t flags,
                                                                const NVCVBorderType    borderMode,
                                                                const pyarray          &borderValue,
                                                                std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    size_t bValueSize = borderValue.size();
    size_t bValueDims = borderValue.ndim();
    if (bValueSize > 4 || bValueDims != 1)
    {
        throw std::runtime_error(FormatString(
            "Channels of borderValue should <= 4 and dimension should be 2, current is '%lu', '%lu' respectively",
            bValueSize, bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nv::cv::cuda::GetElement(bValue, i) = bValueSize > i ? *borderValue.data(i) : 0.f;
    }

    auto warpPerspective = CreateOperator<cvop::WarpPerspective>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, xform});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*warpPerspective});

    warpPerspective->submit(pstream->handle(), input.impl(), output.impl(), xform.impl(), flags, borderMode, bValue);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> WarpPerspectiveVarShape(ImageBatchVarShape &input, Tensor &xform,
                                                            const int32_t flags, const NVCVBorderType borderMode,
                                                            const pyarray &borderValue, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return WarpPerspectiveVarShapeInto(input, *output, xform, flags, borderMode, borderValue, pstream);
}

} // namespace

void ExportOpWarpPerspective(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("warp_perspective", &WarpPerspective, "xform"_a, "flags"_a, py::kw_only(), "border_mode"_a,
                           "border_value"_a, "stream"_a = nullptr);

    DefClassMethod<Tensor>("warp_perspective_into", &WarpPerspectiveInto, "output"_a, "xform"_a, "flags"_a,
                           py::kw_only(), "border_mode"_a, "border_value"_a, "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("warp_perspective", &WarpPerspectiveVarShape, "xform"_a, "flags"_a,
                                       py::kw_only(), "border_mode"_a, "border_value"_a, "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("warp_perspective_into", &WarpPerspectiveVarShapeInto, "output"_a, "xform"_a,
                                       "flags"_a, py::kw_only(), "border_mode"_a, "border_value"_a,
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
