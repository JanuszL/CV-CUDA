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
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <nvcv/cuda/math/LinAlg.hpp> // the object of this test
#include <operators/OpWarpPerspective.hpp>

#include <cmath>
#include <random>

namespace nvcv     = nv::cv;
namespace nvcvcuda = nv::cv::cuda;
namespace test     = nv::cv::test;
using namespace nv::cv::cuda;

// #define DBG_WARP_PERSPECTIVE 1

static void printVec(std::vector<uint8_t> &vec, int height, int rowPitch, int bytesPerPixel, std::string name)
{
#if DBG_WARP_PERSPECTIVE
    for (int i = 0; i < bytesPerPixel; i++)
    {
        std::cout << "\nPrint " << name << " for channel: " << i << std::endl;

        for (int k = 0; k < height; k++)
        {
            for (int j = 0; j < static_cast<int>(rowPitch / bytesPerPixel); j++)
            {
                printf("%4d, ", static_cast<int>(vec[k * rowPitch + j * bytesPerPixel + i]));
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
#endif
}

static uint8_t getPixelForPerspectiveTransform(const uint8_t *srcPtr, const int y, const int x, int k, int width,
                                               int height, int srcRowPitch, int elementsPerPixel,
                                               NVCVBorderType borderMode, const float4 borderVal)
{
    int2 coord = {x, y};
    int2 size  = {width, height};
    if (borderMode == NVCV_BORDER_CONSTANT)
    {
        return (x >= 0 && x < width && y >= 0 && y < height) ? srcPtr[y * srcRowPitch + x * elementsPerPixel + k]
                                                             : static_cast<uint8_t>(GetElement(borderVal, k));
    }
    else if (borderMode == NVCV_BORDER_REPLICATE)
    {
        test::ReplicateBorderIndex(coord, size);
        return srcPtr[coord.y * srcRowPitch + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT)
    {
        test::ReflectBorderIndex(coord, size);
        return srcPtr[coord.y * srcRowPitch + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_REFLECT101)
    {
        test::Reflect101BorderIndex(coord, size);
        return srcPtr[coord.y * srcRowPitch + coord.x * elementsPerPixel + k];
    }
    else if (borderMode == NVCV_BORDER_WRAP)
    {
        test::WrapBorderIndex(coord, size);
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

static void WarpPerspectiveGold(std::vector<uint8_t> &hDst, const int dstRowPitch, const nvcv::Size2D dstSize,
                                const std::vector<uint8_t> &hSrc, const int srcRowPitch, const nvcv::Size2D srcSize,
                                const nvcv::ImageFormat fmt, const NVCVPerspectiveTransform transMatrix,
                                const int flags, const NVCVBorderType borderMode, const float4 borderVal)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    uint8_t       *dstPtr = hDst.data();
    const uint8_t *srcPtr = hSrc.data();

    int srcWidth  = srcSize.w;
    int srcHeight = srcSize.h;

    const int interpolation = flags & NVCV_INTERP_MAX;

    NVCVPerspectiveTransform finalTransformMatrix;

    if (flags & NVCV_WARP_INVERSE_MAP)
    {
        nv::cv::cuda::math::Matrix<float, 3, 3> tempMatrixForInverse;

        tempMatrixForInverse[0][0] = (float)(transMatrix[0]);
        tempMatrixForInverse[0][1] = (float)(transMatrix[1]);
        tempMatrixForInverse[0][2] = (float)(transMatrix[2]);
        tempMatrixForInverse[1][0] = (float)(transMatrix[3]);
        tempMatrixForInverse[1][1] = (float)(transMatrix[4]);
        tempMatrixForInverse[1][2] = (float)(transMatrix[5]);
        tempMatrixForInverse[2][0] = (float)(transMatrix[6]);
        tempMatrixForInverse[2][1] = (float)(transMatrix[7]);
        tempMatrixForInverse[2][2] = (float)(transMatrix[8]);

        math::inv_inplace(tempMatrixForInverse);

        finalTransformMatrix[0] = tempMatrixForInverse[0][0];
        finalTransformMatrix[1] = tempMatrixForInverse[0][1];
        finalTransformMatrix[2] = tempMatrixForInverse[0][2];
        finalTransformMatrix[3] = tempMatrixForInverse[1][0];
        finalTransformMatrix[4] = tempMatrixForInverse[1][1];
        finalTransformMatrix[5] = tempMatrixForInverse[1][2];
        finalTransformMatrix[6] = tempMatrixForInverse[2][0];
        finalTransformMatrix[7] = tempMatrixForInverse[2][1];
        finalTransformMatrix[8] = tempMatrixForInverse[2][2];
    }
    else
    {
        for (int i = 0; i < 9; i++)
        {
            finalTransformMatrix[i] = transMatrix[i];
        }
    }

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            float coeff
                = 1.0f
                / (float)(dst_x * finalTransformMatrix[6] + dst_y * finalTransformMatrix[7] + finalTransformMatrix[8]);
            float src_x
                = coeff
                * (float)(dst_x * finalTransformMatrix[0] + dst_y * finalTransformMatrix[1] + finalTransformMatrix[2]);
            float src_y
                = coeff
                * (float)(dst_x * finalTransformMatrix[3] + dst_y * finalTransformMatrix[4] + finalTransformMatrix[5]);

            if (interpolation == NVCV_INTERP_LINEAR)
            {
                const int x1 = std::floor(src_x);
                const int y1 = std::floor(src_y);

                const int x2 = x1 + 1;
                const int y2 = y1 + 1;

                for (int k = 0; k < elementsPerPixel; k++)
                {
                    float out = 0;

                    uint8_t src_reg = getPixelForPerspectiveTransform(
                        srcPtr, y1, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel, borderMode, borderVal);
                    out += src_reg * ((x2 - src_x) * (y2 - src_y));

                    src_reg = getPixelForPerspectiveTransform(srcPtr, y1, x2, k, srcWidth, srcHeight, srcRowPitch,
                                                              elementsPerPixel, borderMode, borderVal);
                    out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

                    src_reg = getPixelForPerspectiveTransform(srcPtr, y2, x1, k, srcWidth, srcHeight, srcRowPitch,
                                                              elementsPerPixel, borderMode, borderVal);
                    out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

                    src_reg = getPixelForPerspectiveTransform(srcPtr, y2, x2, k, srcWidth, srcHeight, srcRowPitch,
                                                              elementsPerPixel, borderMode, borderVal);
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
                    uint8_t src_reg = getPixelForPerspectiveTransform(
                        srcPtr, y1, x1, k, srcWidth, srcHeight, srcRowPitch, elementsPerPixel, borderMode, borderVal);
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
                            const float w = calcBicubicCoeff(src_x - cx) * calcBicubicCoeff(src_y - cy);
                            uint8_t     src_reg
                                = getPixelForPerspectiveTransform(srcPtr, cy, cx, k, srcWidth, srcHeight, srcRowPitch,
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

/*
    The perspective transform matrix with non-trivial projection are calculated using the below formula:

    input_pts[0] = [0, 0];
    input_pts[1] = [cols - 1, 0];
    input_pts[2] = [0, rows - 1];
    input_pts[3] = [cols - 1, rows - 1];

    output_pts[0] = [0, out_rows*0.13];
    output_pts[1] = [out_cols*0.9, 0];
    output_pts[2] = [out_cols*0.2, out_rows*0.7];
    output_pts[3] = [out_cols*0.8, out_rows];
*/

// clang-format off
NVCV_TEST_SUITE_P(OpWarpPerspective, test::ValueList<int, int, int, int, float, float, float, float, float, float, float, float, float, NVCVInterpolationType, NVCVBorderType, float, float, float, float, int, bool>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,     transformation_matrix,                                         interpolation,              borderType,  borderValue, batchSize, inverse
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_NEAREST,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},

    //------------------ LINEAR INTERP ------------------//
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,      0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5, 0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00,      NVCV_INTERP_LINEAR,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},

    //------------------ CUBIC INTERP ------------------//
    // vary transformation matrix and border type
    {         5,         4,        5,         4,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        5,         4,    0.50, 0.47, 0.00, -0.13, 1.14, 0.52, -0.14, 0.14, 1.00,   NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to larger image
    {         5,         4,        6,         8,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         5,         4,        6,         8,      0.60, 0.56, 0.00, -0.26, 2.28, 1.04, -0.14, 0.14, 1.00, NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},


    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, false},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, false},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, false},

    // change output size to smaller image
    {         7,         8,        4,         5,          1, 0, 0, 0, 1, 0, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,          1, 0, 1, 0, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,          1, 2, 1, 2, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,     0.5, 2, 1, 0.75, 1, 2, 0, 0, 1,                          NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,    NVCV_BORDER_CONSTANT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,   NVCV_BORDER_REPLICATE,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,  NVCV_BORDER_REFLECT101,   1, 2, 3, 4,         4, true},
 // {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,     NVCV_BORDER_REFLECT,   1, 2, 3, 4,         4, true},
    {         7,         8,        4,         5,      0.27, 0.16, 0.00, -0.11, 0.61, 0.65, -0.09, 0.06, 1.00, NVCV_INTERP_CUBIC,        NVCV_BORDER_WRAP,   1, 2, 3, 4,         4, true},
});

// clang-format on

TEST_P(OpWarpPerspective, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVPerspectiveTransform transMatrix;
    transMatrix[0] = GetParamValue<4>();
    transMatrix[1] = GetParamValue<5>();
    transMatrix[2] = GetParamValue<6>();
    transMatrix[3] = GetParamValue<7>();
    transMatrix[4] = GetParamValue<8>();
    transMatrix[5] = GetParamValue<9>();
    transMatrix[6] = GetParamValue<10>();
    transMatrix[7] = GetParamValue<11>();
    transMatrix[8] = GetParamValue<12>();

    NVCVInterpolationType interpolation = GetParamValue<13>();

    NVCVBorderType borderMode = GetParamValue<14>();

    const float4 borderValue = {GetParamValue<15>(), GetParamValue<16>(), GetParamValue<17>(), GetParamValue<18>()};

    int numberOfImages = GetParamValue<19>();

    bool inverseMap = GetParamValue<20>();

    const nvcv::ImageFormat fmt           = nvcv::FMT_RGBA8;
    const int               bytesPerPixel = 4;

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

    nv::cvop::WarpPerspective warpPerspectiveOp;
    EXPECT_NO_THROW(warpPerspectiveOp(stream, imgSrc, imgDst, transMatrix, flags, borderMode, borderValue));

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
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

        // Generate gold result
        WarpPerspectiveGold(goldVec, dstVecRowPitch, {dstWidth, dstHeight}, srcVec[i], srcVecRowPitch,
                            {srcWidth, srcHeight}, fmt, transMatrix, flags, borderMode, borderValue);

        printVec(srcVec[i], srcHeight, srcVecRowPitch, bytesPerPixel, "src vec");
        printVec(goldVec, dstHeight, dstVecRowPitch, bytesPerPixel, "golden output");
        printVec(testVec, dstHeight, dstVecRowPitch, bytesPerPixel, "warped output");

        EXPECT_EQ(goldVec, testVec);
    }
}
