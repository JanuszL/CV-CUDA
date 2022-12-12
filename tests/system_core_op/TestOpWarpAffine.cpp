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

#include "Definitions.hpp"

#include <common/BorderUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpWarpAffine.hpp>

#include <cmath>
#include <random>

namespace nvcv     = nv::cv;
namespace nvcvcuda = nv::cv::cuda;
namespace test     = nv::cv::test;
using namespace nv::cv::cuda;
using namespace test;

//#define DBG 1

template<typename T>
static T getPixel(const T *srcPtr, const int y, const int x, int k, int width, int height, int srcRowPitch,
                  int elementsPerPixel, NVCVBorderType borderMode, const float4 borderVal)
{
    int2 coord = {x, y};
    int2 size  = {width, height};
    if (borderMode == NVCV_BORDER_CONSTANT)
    {
        return (x >= 0 && x < width && y >= 0 && y < height) ? srcPtr[y * srcRowPitch + x * elementsPerPixel + k]
                                                             : static_cast<T>(GetElement(borderVal, k));
    }
    else if (borderMode == NVCV_BORDER_REPLICATE)
    {
        ReplicateBorderIndex(coord, size);
        return srcPtr[coord.y * srcRowPitch + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT)
    {
        ReflectBorderIndex(coord, size);
        return srcPtr[coord.y * srcRowPitch + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT101)
    {
        Reflect101BorderIndex(coord, size);
        return srcPtr[coord.y * srcRowPitch + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_WRAP)
    {
        WrapBorderIndex(coord, size);
        return srcPtr[coord.y * srcRowPitch + coord.x * elementsPerPixel + k];
    }
    else
    {
        return 0;
    }
}

inline float calcBicubicCoeff(float x_)
{
    float x = std::abs(x_);
    if (x <= 1.0f)
    {
        return x * x * (1.5f * x - 2.5f) + 1.0f;
    }
    else if (x < 2.0f)
    {
        return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
    }
    else
    {
        return 0.0f;
    }
}

static void invertAffineTransform(const NVCVAffineTransform xform, NVCVAffineTransform inverseXform)
{
    float den       = xform[0] * xform[4] - xform[1] * xform[3];
    den             = std::abs(den) > 1e-5 ? 1. / den : .0;
    inverseXform[0] = (float)xform[5] * den;
    inverseXform[1] = (float)-xform[1] * den;
    inverseXform[2] = (float)(xform[1] * xform[5] - xform[4] * xform[2]) * den;
    inverseXform[3] = (float)-xform[3] * den;
    inverseXform[4] = (float)xform[0] * den;
    inverseXform[5] = (float)(xform[3] * xform[2] - xform[0] * xform[5]) * den;
}

template<typename T>
static void WarpAffineGold(std::vector<uint8_t> &hDst, int dstRowPitch, nvcv::Size2D dstSize,
                           const std::vector<uint8_t> &hSrc, int srcRowPitch, nvcv::Size2D srcSize,
                           nvcv::ImageFormat fmt, const NVCVAffineTransform xform, const int flags,
                           NVCVBorderType borderMode, const float4 borderVal)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    int srcWidth  = srcSize.w;
    int srcHeight = srcSize.h;

    const int interpolation = flags & NVCV_INTERP_MAX;

    NVCVAffineTransform xform1;

    if (flags & NVCV_WARP_INVERSE_MAP)
    {
        invertAffineTransform(xform, xform1);
    }
    else
    {
        for (int i = 0; i < 6; i++)
        {
            xform1[i] = xform[i];
        }
    }

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            float src_x = (float)(dst_x * xform1[0] + dst_y * xform1[1] + xform1[2]);
            float src_y = (float)(dst_x * xform1[3] + dst_y * xform1[4] + xform1[5]);

            if (interpolation == NVCV_INTERP_LINEAR)
            {
                const int x1 = std::floor(src_x);
                const int y1 = std::floor(src_y);

                const int x2 = x1 + 1;
                const int y2 = y1 + 1;

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    float out = 0;

                    T src_reg = getPixel<T>(srcPtr, y1, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                            borderMode, borderVal);
                    out += src_reg * ((x2 - src_x) * (y2 - src_y));

                    src_reg = getPixel<T>(srcPtr, y1, x2, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                          borderMode, borderVal);
                    out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

                    src_reg = getPixel<T>(srcPtr, y2, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                          borderMode, borderVal);
                    out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

                    src_reg = getPixel<T>(srcPtr, y2, x2, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                          borderMode, borderVal);
                    out     = out + src_reg * ((src_x - x1) * (src_y - y1));

                    out                                                        = std::rint(out);
                    dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k] = out < 0 ? 0 : (out > 255 ? 255 : out);
                }
            }
            else if (interpolation == NVCV_INTERP_NEAREST)
            {
                const int x1 = std::trunc(src_x);
                const int y1 = std::trunc(src_y);
                for (int k = 0; k < elementsPerPixel; k++)
                {
                    T src_reg = getPixel<T>(srcPtr, y1, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel,
                                            borderMode, borderVal);
                    dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k] = src_reg;
                }
            }
            else if (interpolation == NVCV_INTERP_CUBIC)
            {
                const int xmin = std::ceil(src_x - 2.0f);
                const int xmax = std::floor(src_x + 2.0f);

                const int ymin = std::ceil(src_y - 2.0f);
                const int ymax = std::floor(src_y + 2.0f);

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    float sum  = 0;
                    float wsum = 0;

                    for (int cy = ymin; cy <= ymax; cy += 1)
                    {
                        for (int cx = xmin; cx <= xmax; cx += 1)
                        {
                            const float w       = calcBicubicCoeff(src_x - cx) * calcBicubicCoeff(src_y - cy);
                            T           src_reg = getPixel<T>(srcPtr, cy, cx, k, srcWidth, srcHeight, srcRowPitch,
                                                    elementsPerPixel, borderMode, borderVal);
                            sum += w * src_reg;
                            wsum += w;
                        }
                    }

                    float res                                                  = (!wsum) ? 0 : sum / wsum;
                    res                                                        = std::rint(res);
                    dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                }
            }
            else
            {
                return;
            }
        }
    }
}

static std::map<std::vector<int>, std::vector<std::vector<float>>> mapOfTransformationMatrix = {
    {{5, 4, 5, 4}, {{1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, 2}, {1, 2, 1, 2, 1, 2}, {0.5, 2, 1, 0.75, 1, 2}}},
    {{5, 4, 6, 8}, {{1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, 2}, {1, 2, 1, 2, 1, 2}, {0.5, 2, 1, 0.75, 1, 2}}},
    {{7, 8, 4, 5}, {{1, 0, 0, 0, 1, 0}, {1, 0, 1, 0, 1, 2}, {1, 2, 1, 2, 1, 2}, {0.5, 2, 1, 0.75, 1, 2}}}
};

// clang-format off
NVCV_TEST_SUITE_P(OpWarpAffine, test::ValueList<int, int, int, int, float, float, float, float, float, float, NVCVInterpolationType, NVCVBorderType, float, float, float, float, int, bool>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,     transformation_matrix,       interpolation,              borderType,  borderValue, batchSize, inverseAffine
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 2, 0, 1, 1, 0, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 2, 0, 1, 1, 0, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},

    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},

    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},

    // LINEAR INTERP
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 2, 0, 2, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 2, 0, 1, 1, 0,  NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 2, 0, 2, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 2, 0, 2, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 1,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},

    // number of images in batch
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         1,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         8,         false},
    {         7,         8,        4,         5,          2, 2, 1, 3, 1, 2,  NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,        16,         false},

    // CUBIC INTERP
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          1, 2, 0, 2, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         5,         4,        5,         4,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          1, 2, 0, 2, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         5,         4,        6,         8,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          1, 2, 0, 2, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,         false},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,         false},

    // inverse warp affine
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4,          true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4,          true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4,          true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4,          true},
    {         7,         8,        4,         5,          2, 2, 0, 3, 1, 0,   NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4,          true},

});

// clang-format on

TEST_P(OpWarpAffine, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    const NVCVAffineTransform xform = {GetParamValue<4>(), GetParamValue<5>(), GetParamValue<6>(),
                                       GetParamValue<7>(), GetParamValue<8>(), GetParamValue<9>()};

    NVCVInterpolationType interpolation = GetParamValue<10>();

    NVCVBorderType borderMode = GetParamValue<11>();

    const float4 borderValue = {GetParamValue<12>(), GetParamValue<13>(), GetParamValue<14>(), GetParamValue<15>()};

    int numberOfImages = GetParamValue<16>();

    bool inverseMap = GetParamValue<17>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowPitch = srcWidth * fmt.planePixelStrideBytes(0);

    std::default_random_engine randEng;

    for (int i = 0; i < numberOfImages; ++i)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcVecRowPitch);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowPitchBytes(), srcVec[i].data(), srcVecRowPitch,
                               srcVecRowPitch, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    nv::cvop::WarpAffine warpAffineOp(0);
    EXPECT_NO_THROW(warpAffineOp(stream, imgSrc, imgDst, xform, flags, borderMode, borderValue));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowPitch = dstWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(dstHeight * dstVecRowPitch);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowPitch, dstAccess->sampleData(i), dstAccess->rowPitchBytes(),
                               dstVecRowPitch, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstVecRowPitch);

        // Generate gold result
        WarpAffineGold<uint8_t>(goldVec, dstVecRowPitch, {dstWidth, dstHeight}, srcVec[i], srcVecRowPitch,
                                {srcWidth, srcHeight}, fmt, xform, flags, borderMode, borderValue);

#if DBG
        std::cout << "\nPrint src vec " << std::endl;
        for (int k = 0; k < srcHeight; k++)
        {
            for (int j = 0; j < srcVecRowPitch; j++)
            {
                std::cout << static_cast<int>(srcVec[i][k * srcVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowPitch; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint warped output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowPitch; j++)
            {
                std::cout << static_cast<int>(testVec[k * dstVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }
#endif

        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpWarpAffine, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    std::vector<float> transMatrix;
    transMatrix.resize(6);
    transMatrix[0] = GetParamValue<4>();
    transMatrix[1] = GetParamValue<5>();
    transMatrix[2] = GetParamValue<6>();
    transMatrix[3] = GetParamValue<7>();
    transMatrix[4] = GetParamValue<8>();
    transMatrix[5] = GetParamValue<9>();

    NVCVInterpolationType interpolation = GetParamValue<10>();

    NVCVBorderType borderMode = GetParamValue<11>();

    const float4 borderValue = {GetParamValue<12>(), GetParamValue<13>(), GetParamValue<14>(), GetParamValue<15>()};

    int numberOfImages = GetParamValue<16>();

    bool inverseMap = GetParamValue<17>();

    const nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    const int flags = interpolation | (inverseMap ? NVCV_WARP_INVERSE_MAP : 0);

    nvcv::Tensor transMatrixTensor(nvcv::TensorShape({numberOfImages, 6}, nvcv::TensorLayout::NW), nvcv::TYPE_F32);
    const auto  *transMatrixTensorData
        = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(transMatrixTensor.exportData());
    ASSERT_NE(nullptr, transMatrixTensorData);

    auto transMatrixTensorDataAccess = nvcv::TensorDataAccessPitch::Create(*transMatrixTensorData);
    ASSERT_TRUE(transMatrixTensorDataAccess);

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndInputDimsIndex(0, mapOfTransformationMatrix.size() - 1);
    std::uniform_int_distribution<int> rndTransformationMatrixIndex(0, 3);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc, imgDst;
    std::vector<std::vector<float>>           transMatrixHostVec;
    transMatrixHostVec.resize(numberOfImages);

    // List the keys from the map for easy access
    std::vector<std::vector<int>> keysOfMapOfTransformationMatrix;
    for (auto &[key, value] : mapOfTransformationMatrix)
    {
        keysOfMapOfTransformationMatrix.push_back(key);
    }

    for (int i = 0; i < numberOfImages; ++i)
    {
        int tmpSrcWidth  = srcWidth;
        int tmpSrcHeight = srcHeight;

        int tmpDstWidth  = dstWidth;
        int tmpDstHeight = dstHeight;

        std::vector<float> tmpTransMatrix(transMatrix);

        int dictInputIndex          = rndInputDimsIndex(randEng);
        int dictTransformationIndex = rndTransformationMatrixIndex(randEng);

        std::vector<int>   key                        = keysOfMapOfTransformationMatrix[dictInputIndex];
        std::vector<float> chosenTransformationMatrix = mapOfTransformationMatrix[key][dictTransformationIndex];
        // Legacy Reflect & Reflect101 has a bug. So, do special thing for them
        if (i > 0 && !(borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_REFLECT101))
        {
            tmpSrcWidth  = key[0];
            tmpSrcHeight = key[1];

            tmpDstWidth  = key[2];
            tmpDstHeight = key[3];

            tmpTransMatrix = chosenTransformationMatrix;
        }

        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{tmpSrcWidth, tmpSrcHeight}, fmt));

        imgDst.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{tmpDstWidth, tmpDstHeight}, fmt));

        transMatrixHostVec[i] = tmpTransMatrix;

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(transMatrixTensorDataAccess->sampleData(i),
                                    transMatrixTensorDataAccess->samplePitchBytes(), transMatrixHostVec[i].data(),
                                    sizeof(float) * 6, sizeof(float) * 6, 1, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    std::vector<int>                  srcVecRowPitch(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto *srcData = dynamic_cast<const nvcv::IImageDataPitchDevice *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        int srcRowPitch = srcWidth * fmt.planePixelStrideBytes(0);

        srcVecRowPitch[i] = srcRowPitch;

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcRowPitch);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcData->plane(0).buffer, srcData->plane(0).pitchBytes, srcVec[i].data(), srcRowPitch,
                               srcRowPitch, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nv::cvop::WarpAffine warpAffineOp(numberOfImages);
    EXPECT_NO_THROW(warpAffineOp(stream, batchSrc, batchDst, transMatrixTensor, flags, borderMode, borderValue));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nvcv::IImageDataPitchDevice *>(imgSrc[i]->exportData());
        assert(srcData->numPlanes() == 1);
        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        const auto *dstData = dynamic_cast<const nvcv::IImageDataPitchDevice *>(imgDst[i]->exportData());
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int srcRowPitch = srcWidth * fmt.planePixelStrideBytes(0);
        int dstRowPitch = dstWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowPitch);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowPitch, dstData->plane(0).buffer, dstData->plane(0).pitchBytes,
                               dstRowPitch, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowPitch);
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

        NVCVAffineTransform transMatrixForGold;
        transMatrixForGold[0] = transMatrixHostVec[i][0];
        transMatrixForGold[1] = transMatrixHostVec[i][1];
        transMatrixForGold[2] = transMatrixHostVec[i][2];
        transMatrixForGold[3] = transMatrixHostVec[i][3];
        transMatrixForGold[4] = transMatrixHostVec[i][4];
        transMatrixForGold[5] = transMatrixHostVec[i][5];

        // Generate gold result
        WarpAffineGold<uint8_t>(goldVec, dstRowPitch, {dstWidth, dstHeight}, srcVec[i], srcRowPitch,
                                {srcWidth, srcHeight}, fmt, transMatrixForGold, flags, borderMode, borderValue);

#if DBG
        std::cout << "\nPrint src vec " << std::endl;
        for (int k = 0; k < srcHeight; k++)
        {
            for (int j = 0; j < srcVecRowPitch; j++)
            {
                std::cout << static_cast<int>(srcVec[i][k * srcVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowPitch; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint warped output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowPitch; j++)
            {
                std::cout << static_cast<int>(testVec[k * dstVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }
#endif

        EXPECT_EQ(goldVec, testVec);
    }
}
