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
#include <operators/OpGaussian.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> GaussianInto(Tensor &input, Tensor &output, const std::tuple<int, int> &kernel_size,
                                     const std::tuple<double, double> &sigma, NVCVBorderType border,
                                     std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    double2 sigmaArg{std::get<0>(sigma), std::get<1>(sigma)};

    cv::Size2D kernelSizeArg{std::get<0>(kernel_size), std::get<1>(kernel_size)};

    cvop::Gaussian gaussian(kernelSizeArg, 0);

    ResourceGuard roGuard(*pstream, LOCK_READ, {input});
    ResourceGuard rwGuard(*pstream, LOCK_WRITE, {output});

    gaussian(pstream->handle(), input.impl(), output.impl(), kernelSizeArg, sigmaArg, border);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Gaussian(Tensor &input, const std::tuple<int, int> &kernel_size,
                                 const std::tuple<double, double> &sigma, NVCVBorderType border,
                                 std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return GaussianInto(input, *output, kernel_size, sigma, border, pstream);
}

} // namespace

void ExportOpGaussian(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("gaussian", &Gaussian, "kernel_size"_a, "sigma"_a,
                           "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("gaussian_into", &GaussianInto, "output"_a, "kernel_size"_a, "sigma"_a,
                           "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
}

} // namespace nv::cvpy
