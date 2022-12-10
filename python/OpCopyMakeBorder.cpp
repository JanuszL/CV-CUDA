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

#include <nvcv/PixelType.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpCopyMakeBorder.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> CopyMakeBorderInto(Tensor &input, Tensor &output, NVCVBorderType borderMode,
                                           const std::vector<float> &borderValue, int top, int left,
                                           std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    size_t bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw std::runtime_error(FormatString("Channels of borderValue should <= 4, current is '%lu'", bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nv::cv::cuda::GetElement(bValue, i) = bValueDims > i ? borderValue[i] : 0.f;
    }

    auto copyMakeBorder = CreateOperator<cvop::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*copyMakeBorder});

    copyMakeBorder->submit(pstream->handle(), input.impl(), output.impl(), top, left, borderMode, bValue);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> CopyMakeBorder(Tensor &input, NVCVBorderType borderMode, const std::vector<float> &borderValue,
                                       int top, int bottom, int left, int right, std::shared_ptr<Stream> pstream)
{
    Shape in_shape = input.shape();
    Shape out_shape(in_shape);
    int   cdim = out_shape.size() - 1;
    out_shape[cdim - 2] += top + bottom;
    out_shape[cdim - 1] += left + right;

    std::shared_ptr<Tensor> output = Tensor::Create(out_shape, input.dtype(), input.layout());

    return CopyMakeBorderInto(input, *output, borderMode, borderValue, top, left, pstream);
}

std::shared_ptr<Tensor> VarShapeCopyMakeBorderStackInto(ImageBatchVarShape &input, Tensor &output,
                                                        NVCVBorderType            borderMode,
                                                        const std::vector<float> &borderValue, Tensor &top,
                                                        Tensor &left, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    size_t bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw std::runtime_error(FormatString("Channels of borderValue should <= 4, current is '%lu'", bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nv::cv::cuda::GetElement(bValue, i) = bValueDims > i ? borderValue[i] : 0.f;
    }

    auto copyMakeBorder = CreateOperator<cvop::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, top, left});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*copyMakeBorder});

    copyMakeBorder->submit(pstream->handle(), input.impl(), output.impl(), top.impl(), left.impl(), borderMode, bValue);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> VarShapeCopyMakeBorderStack(ImageBatchVarShape &input, NVCVBorderType borderMode,
                                                    const std::vector<float> &borderValue, Tensor &top, Tensor &left,
                                                    int out_height, int out_width, std::shared_ptr<Stream> pstream)
{
    auto format = input.impl().uniqueFormat();
    if (!format)
    {
        throw std::runtime_error(FormatString("All images in input must have the same format."));
    }

    std::shared_ptr<Tensor> output = Tensor::CreateForImageBatch(input.numImages(), {out_width, out_height}, format);

    return VarShapeCopyMakeBorderStackInto(input, *output, borderMode, borderValue, top, left, pstream);
}

std::shared_ptr<ImageBatchVarShape> VarShapeCopyMakeBorderInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                               NVCVBorderType            borderMode,
                                                               const std::vector<float> &borderValue, Tensor &top,
                                                               Tensor &left, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    size_t bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw std::runtime_error(FormatString("Channels of borderValue should <= 4, current is '%lu'", bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++)
    {
        nv::cv::cuda::GetElement(bValue, i) = bValueDims > i ? borderValue[i] : 0.f;
    }

    auto copyMakeBorder = CreateOperator<cvop::CopyMakeBorder>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, top, left});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*copyMakeBorder});

    copyMakeBorder->submit(pstream->handle(), input.impl(), output.impl(), top.impl(), left.impl(), borderMode, bValue);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> VarShapeCopyMakeBorder(ImageBatchVarShape &input, NVCVBorderType borderMode,
                                                           const std::vector<float> &borderValue, Tensor &top,
                                                           Tensor &left, std::vector<int> &out_heights,
                                                           std::vector<int>       &out_widths,
                                                           std::shared_ptr<Stream> pstream)
{
    if (int(out_heights.size()) != input.numImages())
    {
        throw std::runtime_error(
            FormatString("out_heights.size() != input.numImages, %lu != %d", out_heights.size(), input.numImages()));
    }

    if (int(out_widths.size()) != input.numImages())
    {
        throw std::runtime_error(
            FormatString("out_widths.size() != input.numImages, %lu != %d", out_heights.size(), input.numImages()));
    }
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.numImages());

    auto format = input.impl().uniqueFormat();
    if (!format)
    {
        throw std::runtime_error(FormatString("All images in input must have the same format."));
    }

    for (int i = 0; i < input.numImages(); ++i)
    {
        Size2D size = {out_widths[i], out_heights[i]};
        auto   img  = Image::Create(size, format);
        output->pushBack(*img);
    }
    return VarShapeCopyMakeBorderInto(input, *output, borderMode, borderValue, top, left, pstream);
}

} // namespace

void ExportOpCopyMakeBorder(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("copymakeborder", &CopyMakeBorder, "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
                           "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "bottom"_a, "left"_a,
                           "right"_a, "stream"_a = nullptr);
    DefClassMethod<Tensor>(
        "copymakeborder_into", &CopyMakeBorderInto, "output"_a, "border_mode"_a = NVCVBorderType::NVCV_BORDER_CONSTANT,
        "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "left"_a, "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("copymakeborderstack", &VarShapeCopyMakeBorderStack,
                                       "border_mode"_a  = NVCVBorderType::NVCV_BORDER_CONSTANT,
                                       "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "left"_a,
                                       "out_height"_a, "out_width"_a, "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("copymakeborderstack_into", &VarShapeCopyMakeBorderStackInto, "output"_a,
                                       "border_mode"_a  = NVCVBorderType::NVCV_BORDER_CONSTANT,
                                       "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "left"_a,
                                       "stream"_a       = nullptr);
    DefClassMethod<ImageBatchVarShape>("copymakeborder", &VarShapeCopyMakeBorder,
                                       "border_mode"_a  = NVCVBorderType::NVCV_BORDER_CONSTANT,
                                       "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "left"_a,
                                       "out_heights"_a, "out_widths"_a, "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("copymakeborder_into", &VarShapeCopyMakeBorderInto, "output"_a,
                                       "border_mode"_a  = NVCVBorderType::NVCV_BORDER_CONSTANT,
                                       "border_value"_a = std::vector<float>(), py::kw_only(), "top"_a, "left"_a,
                                       "stream"_a       = nullptr);
}

} // namespace nv::cvpy
