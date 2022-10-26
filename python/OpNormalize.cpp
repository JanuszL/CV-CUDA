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

    std::vector<std::shared_ptr<const Resource>> usedResources = {input.shared_from_this(), output.shared_from_this()};

    base.submitSync(*pstream, LOCK_READ);
    scale.submitSync(*pstream, LOCK_READ);
    input.submitSync(*pstream, LOCK_READ);
    output.submitSync(*pstream, LOCK_WRITE);

    if (!flags)
    {
        flags = 0;
    }

    cvop::Normalize normalize;

    normalize(pstream->handle(), input.impl(), base.impl(), scale.impl(), output.impl(), globalScale, globalShift,
              epsilon, *flags);

    try
    {
        base.submitSignal(*pstream, LOCK_READ);
        scale.submitSignal(*pstream, LOCK_READ);
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
