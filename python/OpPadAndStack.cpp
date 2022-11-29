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

    ResourceGuard roGuard(*pstream, LOCK_READ, {input, top, left});
    ResourceGuard rwGuard(*pstream, LOCK_WRITE, {output});

    cvop::PadAndStack padstack;
    padstack(pstream->handle(), input.impl(), output.impl(), top.impl(), left.impl(), border, borderValue);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> PadAndStack(ImageBatchVarShape &input, Tensor &top, Tensor &left, NVCVBorderType border,
                                    float borderValue, std::shared_ptr<Stream> pstream)
{
    cv::ImageFormat fmt = input.impl().uniqueFormat();
    if (fmt == cv::FMT_NONE)
    {
        throw std::runtime_error("All images in the input must have the same format");
    }

    std::shared_ptr<Tensor> output = Tensor::CreateForImageBatch(input.numImages(), input.maxSize(), fmt);

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
