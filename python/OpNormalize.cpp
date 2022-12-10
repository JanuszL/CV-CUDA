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
#include "Tensor.hpp"

#include <operators/OpNormalize.hpp>
#include <pybind11/stl.h>

namespace nv::cvpy {

namespace {

enum OpFlags : uint32_t
{
    SCALE_IS_STDDEV = NVCV_OP_NORMALIZE_SCALE_IS_STDDEV
};

} // namespace

namespace {
std::shared_ptr<Tensor> NormalizeInto(Tensor &input, Tensor &output, Tensor &base, Tensor &scale,
                                      std::optional<uint32_t> flags, float globalScale, float globalShift,
                                      float epsilon, std::shared_ptr<Stream> pstream)
{
    if (pstream == nullptr)
    {
        pstream = Stream::Current().shared_from_this();
    }

    if (!flags)
    {
        flags = 0;
    }

    ResourceGuard guard(*pstream);
    guard.add(LOCK_READ, {input, base, scale});
    guard.add(LOCK_WRITE, {output});

    cvop::Normalize normalize;

    normalize(pstream->handle(), input.impl(), base.impl(), scale.impl(), output.impl(), globalScale, globalShift,
              epsilon, *flags);

    return output.shared_from_this();
}

std::shared_ptr<Tensor> Normalize(Tensor &input, Tensor &base, Tensor &scale, std::optional<uint32_t> flags,
                                  float globalScale, float globalShift, float epsilon, std::shared_ptr<Stream> pstream)
{
    std::shared_ptr<Tensor> output = Tensor::Create(input.shape(), input.dtype(), input.layout());

    return NormalizeInto(input, *output, base, scale, flags, globalScale, globalShift, epsilon, pstream);
}

} // namespace

void ExportOpNormalize(py::module &m)
{
    using namespace pybind11::literals;

    py::enum_<OpFlags>(m, "NormalizeFlags").value("SCALE_IS_STDDEV", OpFlags::SCALE_IS_STDDEV);

    float defGlobalScale = 1;
    float defGlobalShift = 0;
    float defEpsilon     = 0;

    DefClassMethod<Tensor>("normalize", &Normalize, "base"_a, "scale"_a, "flags"_a = std::nullopt, py::kw_only(),
                           "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift, "epsilon"_a = defEpsilon,
                           "stream"_a = nullptr);

    DefClassMethod<Tensor>("normalize_into", &NormalizeInto, "out"_a, "base"_a, "scale"_a, "flags"_a = std::nullopt,
                           py::kw_only(), "globalscale"_a = defGlobalScale, "globalshift"_a = defGlobalShift,
                           "epsilon"_a = defEpsilon, "stream"_a = nullptr);
}

} // namespace nv::cvpy
