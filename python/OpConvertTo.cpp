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
#include "PixelType.hpp"
#include "PyUtil.hpp"
#include "ResourceGuard.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <operators/OpConvertTo.hpp>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> ConvertToInto(Tensor &input, Tensor &output, float scale, float offset,
                                      std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto cvt = CreateOperator<cvop::ConvertTo>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*cvt});

    cvt->submit(pstream->handle(), input.impl(), output.impl(), scale, offset);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> ConvertTo(Tensor &input, cv::PixelType dtype, float scale, float offset,
                                  std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), dtype, input.layout());

    return ConvertToInto(input, *output, scale, offset, pstream);
}

} // namespace

void ExportOpConvertTo(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("convertto", &ConvertTo, "dtype"_a, "scale"_a = 1, "offset"_a = 0, py::kw_only(),
                           "stream"_a = nullptr);
    DefClassMethod<Tensor>("convertto_into", &ConvertToInto, "out"_a, "scale"_a = 1, "offset"_a = 0, py::kw_only(),
                           "stream"_a = nullptr);
}

} // namespace nv::cvpy
