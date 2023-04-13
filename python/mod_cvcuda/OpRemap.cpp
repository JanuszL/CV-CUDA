/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Operators.hpp"

#include <common/PyUtil.hpp>
#include <common/String.hpp>
#include <cvcuda/OpRemap.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/python/ResourceGuard.hpp>
#include <nvcv/python/Stream.hpp>
#include <nvcv/python/Tensor.hpp>

namespace cvcudapy {

namespace {

Tensor RemapInto(Tensor &dst, Tensor &src, Tensor &map, NVCVInterpolationType srcInterp,
                 NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
                 NVCVBorderType borderMode, const pyarray &borderValue, std::optional<Stream> pstream)
{
    if (!pstream)
    {
        pstream = Stream::Current();
    }

    float4 bValue = GetFloat4FromPyArray(borderValue);

    auto op = CreateOperator<cvcuda::Remap>();

    ResourceGuard guard(*pstream);
    guard.add(LockMode::LOCK_READ, {src, map});
    guard.add(LockMode::LOCK_WRITE, {dst});
    guard.add(LockMode::LOCK_NONE, {*op});

    op->submit(pstream->cudaHandle(), src, dst, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode,
               bValue);

    return std::move(dst);
}

Tensor Remap(Tensor &src, Tensor &map, NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
             NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType borderMode,
             const pyarray &borderValue, std::optional<Stream> pstream)
{
    const auto &srcShape = src.shape();
    const auto &mapShape = map.shape();

    if (srcShape.rank() != mapShape.rank())
    {
        throw std::runtime_error("Input src and map tensors must have the same rank");
    }

    Shape dstShape(srcShape.rank());
    for (int i = 0; i < srcShape.rank() - 1; ++i)
    {
        if (mapValueType == NVCV_REMAP_RELATIVE_NORMALIZED)
        {
            dstShape[i] = srcShape[i];
        }
        else
        {
            dstShape[i] = mapShape[i];
        }
    }
    dstShape[srcShape.rank() - 1] = srcShape[srcShape.rank() - 1];

    Tensor dst = Tensor::Create(dstShape, src.dtype(), src.layout());

    return RemapInto(dst, src, map, srcInterp, mapInterp, mapValueType, alignCorners, borderMode, borderValue, pstream);
}

} // namespace

void ExportOpRemap(py::module &m)
{
    using namespace pybind11::literals;

    m.def("remap", &Remap, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST, "map_interp"_a = NVCV_INTERP_NEAREST,
          "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false, "border"_a = NVCV_BORDER_CONSTANT,
          "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr);
    m.def("remap_into", &RemapInto, "dst"_a, "src"_a, "map"_a, "src_interp"_a = NVCV_INTERP_NEAREST,
          "map_interp"_a = NVCV_INTERP_NEAREST, "map_type"_a = NVCV_REMAP_ABSOLUTE, "align_corners"_a = false,
          "border"_a = NVCV_BORDER_CONSTANT, "border_value"_a = pyarray{}, py::kw_only(), "stream"_a = nullptr);
}

} // namespace cvcudapy
