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
#include <operators/OpComposite.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> CompositeInto(Tensor &foreground, Tensor &background, Tensor &fgMask, Tensor &output,
                                      std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    auto composite = CreateOperator<cvop::Composite>();

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {foreground});
    guard.add(LOCK_READ, {background});
    guard.add(LOCK_READ, {fgMask});
    guard.add(LOCK_WRITE, {output});
    guard.add(LOCK_NONE, {*composite});

    composite->submit(pstream->handle(), foreground.impl(), background.impl(), fgMask.impl(), output.impl());

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Composite(Tensor &foreground, Tensor &background, Tensor &fgMask,
                                  std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(foreground.shape(), foreground.dtype(), foreground.layout());

    return CompositeInto(foreground, background, fgMask, *output, pstream);
}

} // namespace

void ExportOpComposite(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("composite", &Composite, "background"_a, "fgMask"_a, py::kw_only(), "stream"_a = nullptr);
    DefClassMethod<Tensor>("composite_into", &CompositeInto, "background"_a, "fgMask"_a, "output"_a, py::kw_only(),
                           "stream"_a = nullptr);
}

} // namespace nv::cvpy
