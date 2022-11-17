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
#include <nvcv/operators/OpMorphology.hpp>
#include <nvcv/operators/Types.h>
#include <nvcv/optools/TypeTraits.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> MorphologyInto(Tensor &input, Tensor &output, NVCVMorphologyType morph_type,
                                       const std::tuple<int, int> &maskSize, const std::tuple<int, int> &anchor,
                                       int32_t iteration, NVCVBorderType border, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto morphology = CreateOperator<cvop::Morphology>(0);

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*morphology});

    cv::Size2D maskSizeArg{std::get<0>(maskSize), std::get<1>(maskSize)};
    int2       anchorArg;
    anchorArg.x = std::get<0>(anchor);
    anchorArg.y = std::get<1>(anchor);

    morphology->submit(pstream->handle(), input.impl(), output.impl(), morph_type, maskSizeArg, anchorArg, iteration,
                       border);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Morphology(Tensor &input, NVCVMorphologyType morph_type, const std::tuple<int, int> &maskSize,
                                   const std::tuple<int, int> &anchor, int32_t iteration, NVCVBorderType border,
                                   std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return MorphologyInto(input, *output, morph_type, maskSize, anchor, iteration, border, pstream);
}

std::shared_ptr<ImageBatchVarShape> MorphologyVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                           NVCVMorphologyType morph_type, Tensor &masks,
                                                           Tensor &anchors, const int32_t iteration,
                                                           const NVCVBorderType    borderMode,
                                                           std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto morphology = CreateOperator<cvop::Morphology>(input.capacity());

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, masks, anchors});
    guard.add(LOCK_WRITE, {output});

    morphology->submit(pstream->handle(), input.impl(), output.impl(), morph_type, masks.impl(), anchors.impl(),
                       iteration, borderMode);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> MorphologyVarShape(ImageBatchVarShape &input, NVCVMorphologyType morph_type,
                                                       Tensor &masks, Tensor &anchors, const int32_t iteration,
                                                       const NVCVBorderType borderMode, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.capacity());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return MorphologyVarShapeInto(input, *output, morph_type, masks, anchors, iteration, borderMode, pstream);
}

} // namespace

void ExportOpMorphology(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("morphology", &Morphology, "morphologyType"_a, "maskSize"_a, "anchor"_a, py::kw_only(),
                           "iteration"_a = 1, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);

    DefClassMethod<Tensor>("morphology_into", &MorphologyInto, "output"_a, "morphologyType"_a, "maskSize"_a, "anchor"_a,
                           py::kw_only(), "iteration"_a = 1, "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
                           "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("morphology", &MorphologyVarShape, "morphologyType"_a, "masks"_a, "anchors"_a,
                                       py::kw_only(), "iteration"_a = 1,
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("morphology_into", &MorphologyVarShapeInto, "output"_a, "morphologyType"_a,
                                       "masks"_a, "anchors"_a, py::kw_only(), "iteration"_a = 1,
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, "stream"_a = nullptr);
}
} // namespace nv::cvpy
