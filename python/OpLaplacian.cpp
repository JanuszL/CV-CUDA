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

#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpLaplacian.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> LaplacianInto(Tensor &input, Tensor &output, const int &ksize, const float &scale,
                                      NVCVBorderType border, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto laplacian = CreateOperator<cvop::Laplacian>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*laplacian});

    laplacian->submit(pstream->handle(), input.impl(), output.impl(), ksize, scale, border);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Laplacian(Tensor &input, const int &ksize, const float &scale, NVCVBorderType border,
                                  std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return LaplacianInto(input, *output, ksize, scale, border, pstream);
}

std::shared_ptr<ImageBatchVarShape> LaplacianVarShapeInto(ImageBatchVarShape &input, ImageBatchVarShape &output,
                                                          Tensor &ksize, Tensor &scale, NVCVBorderType border,
                                                          std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    cvop::Laplacian laplacian;

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, ksize, scale});
    guard.add(LOCK_WRITE, {output});

    laplacian(pstream->handle(), input.impl(), output.impl(), ksize.impl(), scale.impl(), border);

    return output.shared_from_this();
}

std::shared_ptr<ImageBatchVarShape> LaplacianVarShape(ImageBatchVarShape &input, Tensor &ksize, Tensor &scale,
                                                      NVCVBorderType border, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<ImageBatchVarShape> output = ImageBatchVarShape::Create(input.numImages());

    for (int i = 0; i < input.numImages(); ++i)
    {
        cv::ImageFormat format = input.impl()[i].format();
        cv::Size2D      size   = input.impl()[i].size();
        auto            image  = Image::Create(std::tie(size.w, size.h), format);
        output->pushBack(*image);
    }

    return LaplacianVarShapeInto(input, *output, ksize, scale, border, pstream);
}

} // namespace

void ExportOpLaplacian(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("laplacian", &Laplacian, "ksize"_a, "scale"_a = 1.f,
                           "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("laplacian_into", &LaplacianInto, "output"_a, "ksize"_a, "scale"_a = 1.f,
                           "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(), "stream"_a = nullptr);

    DefClassMethod<ImageBatchVarShape>("laplacian", &LaplacianVarShape, "ksize"_a, "scale"_a,
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
                                       "stream"_a = nullptr);
    DefClassMethod<ImageBatchVarShape>("laplacian_into", &LaplacianVarShapeInto, "output"_a, "ksize"_a, "scale"_a,
                                       "border"_a = NVCVBorderType::NVCV_BORDER_CONSTANT, py::kw_only(),
                                       "stream"_a = nullptr);
}

} // namespace nv::cvpy
