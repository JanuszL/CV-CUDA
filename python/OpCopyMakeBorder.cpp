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

#include "Operators.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "String.hpp"
#include "Tensor.hpp"
#include "operators/Types.h"

#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpCopyMakeBorder.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> CopyMakeBorderInto(Tensor &input, Tensor &output, int top, int left, NVCVBorderType borderMode,
                                           const std::vector<float> &borderValue, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    cvop::CopyMakeBorder copyMakeBorder;
    size_t               bValueDims = borderValue.size();
    if (bValueDims > 4)
    {
        throw std::runtime_error(FormatString("Channels of borderValue should <= 4, current is '%lu'", bValueDims));
    }
    float4 bValue;
    for (size_t i = 0; i < 4; i++) nv::cv::cuda::GetElement(bValue, i) = bValueDims > i ? borderValue[i] : 0.f;

    ResourceGuard roGuard(*pstream, LOCK_READ, {input});
    ResourceGuard rwGuard(*pstream, LOCK_WRITE, {output});

    copyMakeBorder(pstream->handle(), input.impl(), output.impl(), top, left, borderMode, bValue);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> CopyMakeBorder(Tensor &input, int top, int buttom, int left, int right,
                                       NVCVBorderType borderMode, const std::vector<float> &borderValue,
                                       std::shared_ptr<Stream> pstream)
{
    Shape in_shape = input.shape();
    Shape out_shape(in_shape);
    int   cdim = out_shape.size() - 1;
    out_shape[cdim - 2] += top + buttom;
    out_shape[cdim - 1] += left + right;

    std::shared_ptr<Tensor> output = Tensor::Create(out_shape, input.dtype(), input.layout());

    return CopyMakeBorderInto(input, *output, top, left, borderMode, borderValue, pstream);
}

} // namespace

void ExportOpCopyMakeBorder(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("copymakeborder", &CopyMakeBorder, "top"_a, "buttom"_a, "left"_a, "right"_a,
                           "border_mode"_a  = NVCVBorderType::NVCV_BORDER_CONSTANT,
                           "border_value"_a = std::vector<float>(), py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("copymakeborder_into", &CopyMakeBorderInto, "output"_a, "top"_a, "left"_a,
                           "border_mode"_a  = NVCVBorderType::NVCV_BORDER_CONSTANT,
                           "border_value"_a = std::vector<float>(), py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
