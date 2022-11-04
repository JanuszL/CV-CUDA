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
#include "Stream.hpp"
#include "Tensor.hpp"

#include <operators/OpPadAndStack.hpp>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> PadAndStackInto(ImageBatchVarShape &input, Tensor &output, Tensor &top, Tensor &left,
                                        NVCVBorderType border, float borderValue, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    std::vector<std::shared_ptr<const Resource>> usedResources
        = {input.shared_from_this(), output.shared_from_this(), top.shared_from_this(), left.shared_from_this()};

    input.submitSync(*pstream, LOCK_READ);
    top.submitSync(*pstream, LOCK_READ);
    left.submitSync(*pstream, LOCK_READ);
    output.submitSync(*pstream, LOCK_WRITE);

    cvop::PadAndStack padstack;

    padstack(pstream->handle(), input.impl(), output.impl(), top.impl(), left.impl(), border, borderValue);

    try
    {
        input.submitSignal(*pstream, LOCK_READ);
        top.submitSignal(*pstream, LOCK_READ);
        left.submitSignal(*pstream, LOCK_READ);
        output.submitSignal(*pstream, LOCK_WRITE);
        pstream->holdResources(std::move(usedResources));
    }
    catch (...)
    {
        pstream->holdResources(std::move(usedResources));
        throw;
    }

    return output.shared_from_this();
}

std::shared_ptr<Tensor> PadAndStack(ImageBatchVarShape &input, Tensor &top, Tensor &left, NVCVBorderType border,
                                    float borderValue, std::shared_ptr<Stream> pstream)
{
    // Lets go through all the images and get the largest dimensions that fit them all
    int32_t maxWidth = 0, maxHeight = 0;
    for (const std::shared_ptr<Image> &img : input)
    {
        maxWidth  = std::max(maxWidth, img->width());
        maxHeight = std::max(maxHeight, img->height());
    }

    std::shared_ptr<Tensor> output
        = Tensor::CreateForImageBatch(input.numImages(), {maxWidth, maxHeight}, input.format());

    return PadAndStackInto(input, *output, top, left, border, borderValue, pstream);
}

} // namespace

void ExportOpPadAndStack(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<ImageBatchVarShape>("padandstack", &PadAndStack, "top"_a, "left"_a,
                                       "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(),
                                       "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("padandstack_into", &PadAndStackInto, "out"_a, "top"_a, "left"_a,
                                       "border"_a = NVCV_BORDER_CONSTANT, "bvalue"_a = 0, py::kw_only(),
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
