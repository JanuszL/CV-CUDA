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
#include "Stream.hpp"
#include "Tensor.hpp"

#include <operators/OpReformat.hpp>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> ReformatInto(Tensor &input, Tensor &output, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    std::vector<std::shared_ptr<const Resource>> usedResources = {input.shared_from_this(), output.shared_from_this()};

    input.submitSync(*pstream, LOCK_READ);
    output.submitSync(*pstream, LOCK_WRITE);

    cvop::Reformat reformat;

    reformat(pstream->handle(), input.impl(), output.impl());

    try
    {
        input.submitSignal(*pstream, LOCK_READ);
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

std::shared_ptr<Tensor> Reformat(Tensor &input, const cv::TensorLayout &out_layout, std::shared_ptr<Stream> pstream)
{
    cv::TensorShape out_shape = Permute(input.impl().shape(), out_layout);

    std::shared_ptr<Tensor> output = Tensor::Create(CreateShape(out_shape), input.dtype(), out_layout);

    return ReformatInto(input, *output, pstream);
}

} // namespace

void ExportOpReformat(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("reformat", &Reformat, "layout"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("reformat_into", &ReformatInto, "out"_a, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
