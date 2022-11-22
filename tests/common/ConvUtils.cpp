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

#include "ConvUtils.hpp"

#include <nvcv/cuda/DropCast.hpp>     // for SaturateCast, etc.
#include <nvcv/cuda/MathOps.hpp>      // for operator *, etc.
#include <nvcv/cuda/SaturateCast.hpp> // for SaturateCast, etc.
#include <nvcv/cuda/TypeTraits.hpp>   // for BaseType, etc.
#include <util/Assert.h>              // for NVCV_ASSERT, etc.

namespace nv::cv::test {

namespace detail {

template<typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, long3 pitches, int b, int y, int x)
{
    return *reinterpret_cast<const T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z]);
}

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, long3 pitches, int b, int y, int x)
{
    return *reinterpret_cast<T *>(&vec[b * pitches.x + y * pitches.y + x * pitches.z]);
}

template<typename T>
inline void Convolve(std::vector<uint8_t> &hDst, const long3 &dstPitches, const std::vector<uint8_t> &hSrc,
                     const long3 &srcPitches, const int3 &shape, const std::vector<float> &kernel,
                     const Size2D &kernelSize, int2 &kernelAnchor, const NVCVBorderType &borderMode,
                     const float4 &borderValue)
{
    using BT  = cuda::BaseType<T>;
    using WT  = cuda::ConvertBaseTypeTo<float, T>;
    int2 size = cuda::DropCast<2>(shape);

    T borderValueT;
    for (int e = 0; e < cuda::NumElements<T>; ++e)
    {
        cuda::GetElement(borderValueT, e) = static_cast<BT>(cuda::GetElement(borderValue, e));
    }

    if (kernelAnchor.x < 0)
    {
        kernelAnchor.x = kernelSize.w / 2;
    }
    if (kernelAnchor.y < 0)
    {
        kernelAnchor.y = kernelSize.h / 2;
    }

    for (int b = 0; b < shape.z; ++b)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                WT res = cuda::SetAll<WT>(0);

                int2 coord;

                for (int ky = 0; ky < kernelSize.h; ++ky)
                {
                    coord.y = y + ky - kernelAnchor.y;

                    for (int kx = 0; kx < kernelSize.w; ++kx)
                    {
                        coord.x = x + kx - kernelAnchor.x;

                        T srcValue = IsInside(coord, size, borderMode)
                                       ? ValueAt<T>(hSrc, srcPitches, b, coord.y, coord.x)
                                       : borderValueT;

                        res += srcValue * kernel[ky * kernelSize.w + kx];
                    }
                }

                ValueAt<T>(hDst, dstPitches, b, y, x) = cuda::SaturateCast<BT>(res);
            }
        }
    }
}

#define NVCV_TEST_INST(TYPE)                                                                                            \
    template const TYPE &ValueAt<TYPE>(const std::vector<uint8_t> &, long3, int, int, int);                             \
    template TYPE       &ValueAt<TYPE>(std::vector<uint8_t> &, long3, int, int, int);                                   \
    template void        Convolve<TYPE>(std::vector<uint8_t> & hDst, const long3 &dstPitches,                           \
                                 const std::vector<uint8_t> &hSrc, const long3 &srcPitches, const int3 &shape,   \
                                 const std::vector<float> &kernel, const Size2D &kernelSize, int2 &kernelAnchor, \
                                 const NVCVBorderType &borderMode, const float4 &borderValue)

NVCV_TEST_INST(uint8_t);
NVCV_TEST_INST(ushort);
NVCV_TEST_INST(uchar3);
NVCV_TEST_INST(uchar4);
NVCV_TEST_INST(float4);
NVCV_TEST_INST(float3);

#undef NVCV_TEST_INST

} // namespace detail

void Convolve(std::vector<uint8_t> &hDst, const long3 &dstPitches, const std::vector<uint8_t> &hSrc,
              const long3 &srcPitches, const int3 &shape, const ImageFormat &format, const std::vector<float> &kernel,
              const Size2D &kernelSize, int2 &kernelAnchor, const NVCVBorderType &borderMode, const float4 &borderValue)
{
    NVCV_ASSERT(format.numPlanes() == 1);

    switch (format.planePixelType(0))
    {
#define NVCV_TEST_CASE(PIXELTYPE, TYPE)                                                                     \
    case NVCV_PIXEL_TYPE_##PIXELTYPE:                                                                       \
        detail::Convolve<TYPE>(hDst, dstPitches, hSrc, srcPitches, shape, kernel, kernelSize, kernelAnchor, \
                               borderMode, borderValue);                                                    \
        break

        NVCV_TEST_CASE(U8, uint8_t);
        NVCV_TEST_CASE(U16, ushort);
        NVCV_TEST_CASE(3U8, uchar3);
        NVCV_TEST_CASE(4U8, uchar4);
        NVCV_TEST_CASE(4F32, float4);
        NVCV_TEST_CASE(3F32, float3);

#undef NVCV_TEST_CASE

    default:
        break;
    }
}

} // namespace nv::cv::test
