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

#include <operators/OpErase.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {
std::shared_ptr<Tensor> EraseInto(Tensor &input, Tensor &output, Tensor &anchor, Tensor &erasing, Tensor &values,
                                  Tensor &imgIdx, bool random, unsigned int seed, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    if (anchor.layout()->ndim() != 1 || (*anchor.layout())[0] != 'N')
    {
        throw std::runtime_error(FormatString("Layout of anchor must be 'N'."));
    }

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, anchor, erasing, values, imgIdx});
    guard.add(LOCK_WRITE, {output});

    Shape       shape = anchor.shape();
    cvop::Erase erase((int)shape[0]);

    erase(pstream->handle(), input.impl(), output.impl(), anchor.impl(), erasing.impl(), values.impl(), imgIdx.impl(),
          random, seed);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Erase(Tensor &input, Tensor &anchor, Tensor &erasing, Tensor &values, Tensor &imgIdx,
                              bool random, unsigned int seed, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return EraseInto(input, *output, anchor, erasing, values, imgIdx, random, seed, pstream);
}

} // namespace

void ExportOpErase(py::module &m)
{
    using namespace pybind11::literals;

    DefClassMethod<Tensor>("erase", &Erase, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a, py::kw_only(),
                           "random"_a = false, "seed"_a = 0, "stream"_a = nullptr);
    DefClassMethod<Tensor>("erase_into", &EraseInto, "out"_a, "anchor"_a, "erasing"_a, "values"_a, "imgIdx"_a,
                           py::kw_only(), "random"_a = false, "seed"_a = 0, "stream"_a = nullptr);
}

} // namespace nv::cvpy
