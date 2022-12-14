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

#include "FlipUtils.hpp"

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
inline void flip(std::vector<uint8_t> &hDst, const long3 &dstPitches, const std::vector<uint8_t> &hSrc,
                 const long3 &srcPitches, const int3 &shape, int flipCode)
{
    using BT  = cuda::BaseType<T>;
    int2 size = cuda::DropCast<2>(shape);

    for (int b = 0; b < shape.z; ++b)
    {
        for (int y = 0; y < shape.y; ++y)
        {
            for (int x = 0; x < shape.x; ++x)
            {
                T srcValue;
                if (flipCode > 0)
                {
                    srcValue = ValueAt<T>(hSrc, srcPitches, b, y, (size.x - 1 - x));
                }
                else if (flipCode == 0)
                {
                    srcValue = ValueAt<T>(hSrc, srcPitches, b, (size.y - 1 - y), x);
                }
                else
                {
                    srcValue = ValueAt<T>(hSrc, srcPitches, b, (size.y - 1 - y), (size.x - 1 - x));
                }

                ValueAt<T>(hDst, dstPitches, b, y, x) = cuda::SaturateCast<BT>(srcValue);
            }
        }
    }
}

#define NVCV_TEST_INST(TYPE)                                                                                         \
    template const TYPE &ValueAt<TYPE>(const std::vector<uint8_t> &, long3, int, int, int);                          \
    template TYPE       &ValueAt<TYPE>(std::vector<uint8_t> &, long3, int, int, int);                                \
    template void flip<TYPE>(std::vector<uint8_t> & hDst, const long3 &dstPitches, const std::vector<uint8_t> &hSrc, \
                             const long3 &srcPitches, const int3 &shape, int flipCode)

NVCV_TEST_INST(uint8_t);
NVCV_TEST_INST(ushort);
NVCV_TEST_INST(uchar3);
NVCV_TEST_INST(uchar4);
NVCV_TEST_INST(float4);
NVCV_TEST_INST(float3);

#undef NVCV_TEST_INST

} // namespace detail

void FlipCPU(std::vector<uint8_t> &hDst, const long3 &dstPitches, const std::vector<uint8_t> &hSrc,
             const long3 &srcPitches, const int3 &shape, const ImageFormat &format, int flipCode)
{
    NVCV_ASSERT(format.numPlanes() == 1);

    switch (format.planePixelType(0))
    {
#define NVCV_TEST_CASE(PIXELTYPE, TYPE)                                          \
    case NVCV_PIXEL_TYPE_##PIXELTYPE:                                            \
        detail::flip<TYPE>(hDst, dstPitches, hSrc, srcPitches, shape, flipCode); \
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
