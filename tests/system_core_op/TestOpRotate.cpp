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

#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpRotate.hpp>

#include <cmath>
#include <random>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;
namespace t    = ::testing;

#define PI 3.1415926535897932384626433832795

// #define DBG_ROTATE 1

static void compute_warpAffine(const double angle, const double xShift, const double yShift, double *aCoeffs)
{
    aCoeffs[0] = cos(angle * PI / 180);
    aCoeffs[1] = sin(angle * PI / 180);
    aCoeffs[2] = xShift;
    aCoeffs[3] = -sin(angle * PI / 180);
    aCoeffs[4] = cos(angle * PI / 180);
    aCoeffs[5] = yShift;
}

static void compute_center_shift(const int center_x, const int center_y, const double angle, double &xShift,
                                 double &yShift)
{
    xShift = (1 - cos(angle * PI / 180)) * center_x - sin(angle * PI / 180) * center_y;
    yShift = sin(angle * PI / 180) * center_x + (1 - cos(angle * PI / 180)) * center_y;
}

static void assignCustomValuesInSrc(std::vector<uint8_t> &srcVec, int srcWidth, int srcHeight, int srcVecRowPitch)
{
    int initialValue = 1;
    int pixelBytes   = static_cast<int>(srcVecRowPitch / srcWidth);
    for (int i = 0; i < srcHeight; i++)
    {
        for (int j = 0; j < srcVecRowPitch; j = j + pixelBytes)
        {
            for (int k = 0; k < pixelBytes; k++)
            {
                srcVec[i * srcVecRowPitch + j + k] = initialValue;
            }
            initialValue++;
        }
    }

#if DBG_ROTATE
    std::cout << "\nPrint input " << std::endl;

    for (int i = 0; i < srcHeight; i++)
    {
        for (int j = 0; j < srcVecRowPitch; j++)
        {
            std::cout << static_cast<int>(srcVec[i * srcVecRowPitch + j]) << ",";
        }
        std::cout << std::endl;
    }
#endif
}

template<typename T>
static void Rotate(std::vector<T> &hDst, int dstRowPitch, nvcv::Size2D dstSize, const std::vector<T> &hSrc,
                   int srcRowPitch, nvcv::Size2D srcSize, nvcv::ImageFormat fmt, const double angleDeg,
                   const double2 shift, NVCVInterpolationType interpolation)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    T       *dstPtr = hDst.data();
    const T *srcPtr = hSrc.data();

    // calculate coefficients
    double d_aCoeffs[6];
    compute_warpAffine(angleDeg, shift.x, shift.y, d_aCoeffs);

    int width  = dstSize.w;
    int height = dstSize.h;

    for (int dst_y = 0; dst_y < dstSize.h; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstSize.w; dst_x++)
        {
            if (interpolation == NVCV_INTERP_LINEAR)
            {
                const double dst_x_shift = dst_x - d_aCoeffs[2];
                const double dst_y_shift = dst_y - d_aCoeffs[5];
                float        src_x       = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
                float        src_y       = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

                if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
                {
                    const int x1 = src_x > 0 ? std::floor(src_x) : std::rint(src_x);
                    const int y1 = src_y > 0 ? std::floor(src_y) : std::rint(src_y);

                    const int x2      = x1 + 1;
                    const int y2      = y1 + 1;
                    const int x2_read = std::min(x2, width - 1);
                    const int y2_read = std::min(y2, height - 1);

                    for (int k = 0; k < elementsPerPixel; k++)
                    {
                        float out = 0.;

                        T src_reg = srcPtr[y1 * srcRowPitch + x1 * elementsPerPixel + k];
                        out       = out + src_reg * ((x2 - src_x) * (y2 - src_y));

                        src_reg = srcPtr[y1 * srcRowPitch + x2_read * elementsPerPixel + k];
                        out     = out + src_reg * ((src_x - x1) * (y2 - src_y));

                        src_reg = srcPtr[y2_read * srcRowPitch + x1 * elementsPerPixel + k];
                        out     = out + src_reg * ((x2 - src_x) * (src_y - y1));

                        src_reg = srcPtr[y2_read * srcRowPitch + x2_read * elementsPerPixel + k];
                        out     = out + src_reg * ((src_x - x1) * (src_y - y1));

                        out = std::rint(out);
                        dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k]
                            = out < 0 ? 0 : (out > 255 ? 255 : out);
                    }
                }
            }
            else if (interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_CUBIC)
            {
                /*
                    Use this for NVCV_INTERP_CUBIC interpolation only for angles - {90, 180}
                */

                const double dst_x_shift = dst_x - d_aCoeffs[2];
                const double dst_y_shift = dst_y - d_aCoeffs[5];

                float src_x = (float)(dst_x_shift * d_aCoeffs[0] + dst_y_shift * (-d_aCoeffs[1]));
                float src_y = (float)(dst_x_shift * (-d_aCoeffs[3]) + dst_y_shift * d_aCoeffs[4]);

                if (src_x > -0.5 && src_x < width && src_y > -0.5 && src_y < height)
                {
                    const int x1 = std::min(static_cast<int>(src_x + 0.5), width - 1);
                    const int y1 = std::min(static_cast<int>(src_y + 0.5), height - 1);

                    for (int k = 0; k < elementsPerPixel; k++)
                    {
                        dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k]
                            = srcPtr[y1 * srcRowPitch + x1 * elementsPerPixel + k];
                    }
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpRotate, test::ValueList<int, int, int, int, NVCVInterpolationType, int, double>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,         interpolation, numberImages, angle
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           1,     90},
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           4,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            1,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            4,     90},

    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           1,     45},
    {         4,         4,        4,         4,    NVCV_INTERP_NEAREST,           4,     45},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            1,     45},
    {         5,         5,        5,         5,    NVCV_INTERP_LINEAR,            4,     45},

    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             1,     90},
    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             4,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             1,     90},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             4,     90},

    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             1,     180},
    {         4,         4,        4,         4,    NVCV_INTERP_CUBIC,             4,     180},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             1,     180},
    {         5,         5,        5,         5,    NVCV_INTERP_CUBIC,             4,     180},
});

// clang-format on

TEST_P(OpRotate, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    double angleDeg = GetParamValue<6>();
    double shiftX   = -1;
    double shiftY   = -1;

    const nvcv::ImageFormat fmt = nvcv::FMT_RGB8;

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowPitch = srcWidth * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < numberOfImages; ++i)
    {
        srcVec[i].resize(srcHeight * srcVecRowPitch);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return 0; });

        // Assign custom values in input vector
        assignCustomValuesInSrc(srcVec[i], srcWidth, srcHeight, srcVecRowPitch);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowPitchBytes(), srcVec[i].data(), srcVecRowPitch,
                               srcVecRowPitch, // vec has no padding
                               srcHeight, cudaMemcpyHostToDevice));
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, fmt);

    // Compute shiftX, shiftY using center
    int center_x = (srcWidth - 1) / 2, center_y = (srcHeight - 1) / 2;
    compute_center_shift(center_x, center_y, angleDeg, shiftX, shiftY);

    nv::cvop::Rotate RotateOp;
    double2          shift = {shiftX, shiftY};
    EXPECT_NO_THROW(RotateOp(stream, imgSrc, imgDst, angleDeg, shift, interpolation));

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
        Rotate<uint8_t>(goldVec, dstVecRowPitch, {dstWidth, dstHeight}, srcVec[i], srcVecRowPitch,
                        {srcWidth, srcHeight}, fmt, angleDeg, shift, interpolation);

#if DBG_ROTATE
        std::cout << "\nPrint golden output " << std::endl;

        for (int k = 0; k < dstHeight; k++)
        {
            for (int j = 0; j < dstVecRowPitch; j++)
            {
                std::cout << static_cast<int>(goldVec[k * dstVecRowPitch + j]) << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "\nPrint rotated output " << std::endl;

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
