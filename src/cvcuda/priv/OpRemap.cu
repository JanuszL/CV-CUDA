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

#include "OpRemap.hpp"

#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorLayout.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/InterpolationWrap.hpp>
#include <nvcv/cuda/MathOps.hpp>
#include <nvcv/cuda/StaticCast.hpp>
#include <util/Math.hpp>

namespace cuda = nvcv::cuda;
namespace util = nvcv::util;

namespace {

template<class SrcWrapper, class DstWrapper, class MapWrapper>
__global__ void remap(SrcWrapper src, DstWrapper dst, MapWrapper map, int2 dstSize, float2 srcScale, float2 mapScale,
                      float2 valScale, float2 srcOffset, float dstOffset)
{
    int3 dstCoord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (dstCoord.x >= dstSize.x || dstCoord.y >= dstSize.y)
    {
        return;
    }

    float3 mapCoord{0.f, 0.f, static_cast<float>(dstCoord.z)};

    // The map is accessed at destination coordinate, with destination offset and scaled by map scale.  The
    // resulting map coordinate is interpolated in the map, given by the map interpolation type.

    mapCoord.x = (dstCoord.x + dstOffset) * mapScale.x;
    mapCoord.y = (dstCoord.y + dstOffset) * mapScale.y;

    float2 mapValue = map[mapCoord];

    float3 srcCoord{0.f, 0.f, static_cast<float>(dstCoord.z)};

    // The source is accessed at destination coordinate scaled by source scale, plus the map value that is either a
    // relative distance from destination or an absolute position at source (either normalized or not), multiplied
    // by value scale and offset by source offset.  The result of the map value scaled must be rounded to get an
    // absolute position regardless of source interpolation.  The source interpolation type only affects the source
    // scaling and offset values.

    srcCoord.x = dstCoord.x * srcScale.x + mapValue.x * valScale.x + srcOffset.x;
    srcCoord.y = dstCoord.y * srcScale.y + mapValue.y * valScale.y + srcOffset.y;

    dst[dstCoord] = src[srcCoord];
}

template<typename T, NVCVBorderType B, NVCVInterpolationType MI, NVCVInterpolationType SI>
void RunRemap(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
              const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &mapData,
              NVCVRemapMapValueType mapValueType, bool alignCorners, const T &borderValue)
{
    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(mapData);

    int2 srcSize = cuda::StaticCast<int>(long2{srcAccess->numCols(), srcAccess->numRows()});
    int2 dstSize = cuda::StaticCast<int>(long2{dstAccess->numCols(), dstAccess->numRows()});
    int2 mapSize = cuda::StaticCast<int>(long2{mapAccess->numCols(), mapAccess->numRows()});

    // To avoid floating-point issues, instead of normalizing coordinates by dividing them by destination size, it
    // is better to compute the {source, map, map value} scale by dividing its size by destination size and use it
    // to scale the {source, map} coordinates accordingly.  The map value affects the source coordinates, and the
    // source offset is used to shift its coordinate position depending on the map value type.
    float2 srcScale, mapScale, valScale, srcOffset;
    float  dstOffset;

    switch (mapValueType)
    {
    case NVCV_REMAP_ABSOLUTE:
        srcScale  = float2{0.f, 0.f};
        mapScale  = cuda::StaticCast<float>(mapSize) / dstSize;
        valScale  = float2{1.f, 1.f};
        srcOffset = float2{0.f, 0.f};
        dstOffset = 0.f;
        break;
    case NVCV_REMAP_ABSOLUTE_NORMALIZED:
        srcScale  = float2{0.f, 0.f};
        mapScale  = cuda::StaticCast<float>(mapSize) / dstSize;
        valScale  = (srcSize - (alignCorners ? 1.f : 0.f)) / 2.f;
        srcOffset = valScale - (alignCorners ? 0.f : .5f);
        dstOffset = 0.f;
        break;
    case NVCV_REMAP_RELATIVE_NORMALIZED:
        srcScale  = cuda::StaticCast<float>(srcSize) / dstSize;
        mapScale  = (mapSize - 1.f) / dstSize;
        valScale  = srcSize - 1.f;
        dstOffset = alignCorners ? 0.f : .5f;
        srcOffset = srcScale * dstOffset - dstOffset;
        break;
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid map value type");
    }

    dim3 block(32, 4, 1);
    dim3 grid(util::DivUp(dstSize.x, block.x), util::DivUp(dstSize.y, block.y), dstAccess->numSamples());

    auto map = cuda::CreateInterpolationWrapNHW<const float2, NVCV_BORDER_REPLICATE, MI>(mapData);
    auto src = cuda::CreateInterpolationWrapNHW<const T, B, SI>(srcData, borderValue);
    auto dst = cuda::CreateTensorWrapNHW<T>(dstData);

    remap<<<grid, block, 0, stream>>>(src, dst, map, dstSize, srcScale, mapScale, valScale, srcOffset, dstOffset);
}

template<typename T, NVCVBorderType B, NVCVInterpolationType MI>
void RunRemap(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
              const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &mapData,
              NVCVInterpolationType srcInterp, NVCVRemapMapValueType mapValueType, bool alignCorners,
              const T &borderValue)
{
#define NVCV_RUN_REMAP(INTERP_TYPE)                                                                                  \
    case NVCV_INTERP_##INTERP_TYPE:                                                                                  \
        RunRemap<T, B, MI, NVCV_INTERP_##INTERP_TYPE>(stream, srcData, dstData, mapData, mapValueType, alignCorners, \
                                                      borderValue);                                                  \
        break

    switch (srcInterp)
    {
        NVCV_RUN_REMAP(NEAREST);
        NVCV_RUN_REMAP(LINEAR);
        NVCV_RUN_REMAP(CUBIC);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid input interpolation type");
    }

#undef NVCV_RUN_REMAP
}

template<typename T, NVCVBorderType B>
void RunRemap(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
              const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &mapData,
              NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType,
              bool alignCorners, const T &borderValue)
{
#define NVCV_RUN_REMAP(INTERP_TYPE)                                                                           \
    case NVCV_INTERP_##INTERP_TYPE:                                                                           \
        RunRemap<T, B, NVCV_INTERP_##INTERP_TYPE>(stream, srcData, dstData, mapData, srcInterp, mapValueType, \
                                                  alignCorners, borderValue);                                 \
        break

    switch (mapInterp)
    {
        NVCV_RUN_REMAP(NEAREST);
        NVCV_RUN_REMAP(LINEAR);
        NVCV_RUN_REMAP(CUBIC);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid map interpolation type");
    }

#undef NVCV_RUN_REMAP
}

template<typename T>
void RunRemap(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
              const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &mapData,
              NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp, NVCVRemapMapValueType mapValueType,
              bool alignCorners, NVCVBorderType border, const float4 &borderValue)
{
    const T bvalue = cuda::DropCast<cuda::NumElements<T>>(cuda::StaticCast<cuda::BaseType<T>>(borderValue));

#define NVCV_RUN_REMAP(BORDER_TYPE)                                                                                   \
    case NVCV_BORDER_##BORDER_TYPE:                                                                                   \
        RunRemap<T, NVCV_BORDER_##BORDER_TYPE>(stream, srcData, dstData, mapData, srcInterp, mapInterp, mapValueType, \
                                               alignCorners, bvalue);                                                 \
        break

    switch (border)
    {
        NVCV_RUN_REMAP(CONSTANT);
        NVCV_RUN_REMAP(REPLICATE);
        NVCV_RUN_REMAP(REFLECT);
        NVCV_RUN_REMAP(WRAP);
        NVCV_RUN_REMAP(REFLECT101);
    default:
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid border type");
    }

#undef NVCV_RUN_REMAP
}

inline void RunRemap(cudaStream_t stream, const nvcv::TensorDataStridedCuda &srcData,
                     const nvcv::TensorDataStridedCuda &dstData, const nvcv::TensorDataStridedCuda &mapData,
                     NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                     NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType border,
                     const float4 &borderValue)
{
    if (srcData.layout() != dstData.layout())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output layout are different");
    }
    if (srcData.dtype() != dstData.dtype())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Input and output data type are different");
    }

    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(dstData);
    int  numChannels{dstAccess->numChannels()};

    // clang-format off

#define NVCV_RUN_REMAP(BT, DT, T)                                                                        \
    ((dstData.dtype() == nvcv::TYPE_##BT && numChannels == cuda::NumElements<T>) ||                      \
     (dstData.dtype() == nvcv::TYPE_##DT && numChannels == 1))                                           \
        RunRemap<T>(stream, srcData, dstData, mapData, srcInterp, mapInterp, mapValueType, alignCorners, \
                    border, borderValue)

    if (dstData.layout() == nvcv::TENSOR_HWC || dstData.layout() == nvcv::TENSOR_NHWC)
    {
        if NVCV_RUN_REMAP(U8, U8, uchar1);
        else if NVCV_RUN_REMAP(U8, 3U8, uchar3);
        else if NVCV_RUN_REMAP(U8, 4U8, uchar4);
        else if NVCV_RUN_REMAP(F32, F32, float1);
        else
        {
            throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid data type in input/output");
        }
    }
    else
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Invalid tensor layout in input/output");
    }

#undef NVCV_RUN_REMAP

    // clang-format on
}

} // anonymous namespace

namespace cvcuda::priv {

Remap::Remap() {}

void Remap::operator()(cudaStream_t stream, nvcv::ITensor &src, nvcv::ITensor &dst, nvcv::ITensor &map,
                       NVCVInterpolationType srcInterp, NVCVInterpolationType mapInterp,
                       NVCVRemapMapValueType mapValueType, bool alignCorners, NVCVBorderType border,
                       float4 borderValue) const
{
    auto srcData = src.exportData<nvcv::TensorDataStridedCuda>();
    if (!srcData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Input must be cuda-accessible, pitch-linear tensor");
    }

    auto dstData = dst.exportData<nvcv::TensorDataStridedCuda>();
    if (!dstData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Output must be cuda-accessible, pitch-linear tensor");
    }

    auto mapData = map.exportData<nvcv::TensorDataStridedCuda>();
    if (!mapData)
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                              "Remap map input must be cuda-accessible, pitch-linear tensor");
    }

    auto srcAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*srcData);
    auto dstAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*dstData);
    auto mapAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*mapData);
    NVCV_ASSERT(srcAccess && dstAccess && mapAccess);

    if (srcAccess->numChannels() != dstAccess->numChannels())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible input/output number of channels");
    }

    if (srcAccess->numSamples() != dstAccess->numSamples() || dstAccess->numSamples() != mapAccess->numSamples())
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Incompatible number of samples in tensors");
    }

    if (!((mapData->dtype() == nvcv::TYPE_2F32 && mapAccess->numChannels() == 1)
          || (mapData->dtype() == nvcv::TYPE_F32 && mapAccess->numChannels() == 2)))
    {
        throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT, "Remap map input must have 2F32 data type");
    }

    RunRemap(stream, *srcData, *dstData, *mapData, srcInterp, mapInterp, mapValueType, alignCorners, border,
             borderValue);
}

} // namespace cvcuda::priv
