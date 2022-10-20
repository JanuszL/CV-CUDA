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
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpResize.hpp>

//#define DEBUG_PRINT_IMAGE
//#define DEBUG_PRINT_DIFF

#include <common/Utils.hpp>

#include <cmath>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;

static void Resize(std::vector<uint8_t> &hDst, const std::vector<uint8_t> &hSrc,
                   const nvcv::ITensorDataPitchDevice *dDstData, const nvcv::ITensorDataPitchDevice *dSrcData,
                   const int interpolation)
{
    ASSERT_NE(nullptr, dDstData);
    ASSERT_NE(nullptr, dSrcData);

    double iScale = static_cast<double>(dSrcData->dims().h) / dDstData->dims().h;
    double jScale = static_cast<double>(dSrcData->dims().w) / dDstData->dims().w;

    EXPECT_EQ(dDstData->numImages(), dSrcData->numImages());
    EXPECT_EQ(dDstData->dtype(), dSrcData->dtype());

    int elementsPerPixel = dDstData->dims().c;
    int dstRowPitch      = dDstData->rowPitchBytes() / sizeof(uint8_t);
    int srcRowPitch      = dSrcData->rowPitchBytes() / sizeof(uint8_t);
    int dstImgPitch      = dDstData->imgPitchBytes() / sizeof(uint8_t);
    int srcImgPitch      = dSrcData->imgPitchBytes() / sizeof(uint8_t);

    uint8_t       *dstPtrTop = hDst.data();
    const uint8_t *srcPtrTop = hSrc.data();

    for (int img = 0; img < dDstData->numImages(); img++)
    {
        uint8_t       *dstPtr = dstPtrTop + dstImgPitch * img;
        const uint8_t *srcPtr = srcPtrTop + srcImgPitch * img;

        for (int di = 0; di < dDstData->dims().h; di++)
        {
            for (int dj = 0; dj < dDstData->dims().w; dj++)
            {
                if (interpolation == NVCV_INTERP_NEAREST)
                {
                    double fi = iScale * di;
                    double fj = jScale * dj;

                    int si = std::floor(fi);
                    int sj = std::floor(fj);

                    si = std::min(si, dSrcData->dims().h - 1);
                    sj = std::min(sj, dSrcData->dims().w - 1);

                    for (int k = 0; k < dDstData->dims().c; k++)
                    {
                        dstPtr[di * dstRowPitch + dj * elementsPerPixel + k]
                            = srcPtr[si * srcRowPitch + sj * elementsPerPixel + k];
                    }
                }
                else if (interpolation == NVCV_INTERP_LINEAR)
                {
                    double fi = iScale * (di + 0.5) - 0.5;
                    double fj = jScale * (dj + 0.5) - 0.5;

                    int si = std::floor(fi);
                    int sj = std::floor(fj);

                    fi -= si;
                    fj -= sj;

                    fj = (sj < 0 || sj >= dSrcData->dims().w - 1) ? 0 : fj;

                    si = std::max(0, std::min(si, dSrcData->dims().h - 2));
                    sj = std::max(0, std::min(sj, dSrcData->dims().w - 2));

                    double iWeights[2] = {1 - fi, fi};
                    double jWeights[2] = {1 - fj, fj};

                    for (int k = 0; k < dDstData->dims().c; k++)
                    {
                        double res = std::rint(srcPtr[(si + 0) * srcRowPitch + (sj + 0) * elementsPerPixel + k]
                                                   * iWeights[0] * jWeights[0]
                                               + srcPtr[(si + 1) * srcRowPitch + (sj + 0) * elementsPerPixel + k]
                                                     * iWeights[1] * jWeights[0]
                                               + srcPtr[(si + 0) * srcRowPitch + (sj + 1) * elementsPerPixel + k]
                                                     * iWeights[0] * jWeights[1]
                                               + srcPtr[(si + 1) * srcRowPitch + (sj + 1) * elementsPerPixel + k]
                                                     * iWeights[1] * jWeights[1]);

                        dstPtr[di * dstRowPitch + dj * elementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                    }
                }
                else if (interpolation == NVCV_INTERP_CUBIC)
                {
                    double fi = iScale * (di + 0.5) - 0.5;
                    double fj = jScale * (dj + 0.5) - 0.5;

                    int si = std::floor(fi);
                    int sj = std::floor(fj);

                    fi -= si;
                    fj -= sj;

                    fj = (sj < 1 || sj >= dSrcData->dims().w - 3) ? 0 : fj;

                    si = std::max(1, std::min(si, dSrcData->dims().h - 3));
                    sj = std::max(1, std::min(sj, dSrcData->dims().w - 3));

                    const double A = -0.75;
                    double       iWeights[4];
                    iWeights[0] = ((A * (fi + 1) - 5 * A) * (fi + 1) + 8 * A) * (fi + 1) - 4 * A;
                    iWeights[1] = ((A + 2) * fi - (A + 3)) * fi * fi + 1;
                    iWeights[2] = ((A + 2) * (1 - fi) - (A + 3)) * (1 - fi) * (1 - fi) + 1;
                    iWeights[3] = 1 - iWeights[0] - iWeights[1] - iWeights[2];

                    double jWeights[4];
                    jWeights[0] = ((A * (fj + 1) - 5 * A) * (fj + 1) + 8 * A) * (fj + 1) - 4 * A;
                    jWeights[1] = ((A + 2) * fj - (A + 3)) * fj * fj + 1;
                    jWeights[2] = ((A + 2) * (1 - fj) - (A + 3)) * (1 - fj) * (1 - fj) + 1;
                    jWeights[3] = 1 - jWeights[0] - jWeights[1] - jWeights[2];

                    for (int k = 0; k < dDstData->dims().c; k++)
                    {
                        double res = std::abs(std::rint(
                            srcPtr[(si - 1) * srcRowPitch + (sj - 1) * elementsPerPixel + k] * jWeights[0] * iWeights[0]
                            + srcPtr[(si + 0) * srcRowPitch + (sj - 1) * elementsPerPixel + k] * jWeights[0]
                                  * iWeights[1]
                            + srcPtr[(si + 1) * srcRowPitch + (sj - 1) * elementsPerPixel + k] * jWeights[0]
                                  * iWeights[2]
                            + srcPtr[(si + 2) * srcRowPitch + (sj - 1) * elementsPerPixel + k] * jWeights[0]
                                  * iWeights[3]
                            + srcPtr[(si - 1) * srcRowPitch + (sj + 0) * elementsPerPixel + k] * jWeights[1]
                                  * iWeights[0]
                            + srcPtr[(si + 0) * srcRowPitch + (sj + 0) * elementsPerPixel + k] * jWeights[1]
                                  * iWeights[1]
                            + srcPtr[(si + 1) * srcRowPitch + (sj + 0) * elementsPerPixel + k] * jWeights[1]
                                  * iWeights[2]
                            + srcPtr[(si + 2) * srcRowPitch + (sj + 0) * elementsPerPixel + k] * jWeights[1]
                                  * iWeights[3]
                            + srcPtr[(si - 1) * srcRowPitch + (sj + 1) * elementsPerPixel + k] * jWeights[2]
                                  * iWeights[0]
                            + srcPtr[(si + 0) * srcRowPitch + (sj + 1) * elementsPerPixel + k] * jWeights[2]
                                  * iWeights[1]
                            + srcPtr[(si + 1) * srcRowPitch + (sj + 1) * elementsPerPixel + k] * jWeights[2]
                                  * iWeights[2]
                            + srcPtr[(si + 2) * srcRowPitch + (sj + 1) * elementsPerPixel + k] * jWeights[2]
                                  * iWeights[3]
                            + srcPtr[(si - 1) * srcRowPitch + (sj + 2) * elementsPerPixel + k] * jWeights[3]
                                  * iWeights[0]
                            + srcPtr[(si + 0) * srcRowPitch + (sj + 2) * elementsPerPixel + k] * jWeights[3]
                                  * iWeights[1]
                            + srcPtr[(si + 1) * srcRowPitch + (sj + 2) * elementsPerPixel + k] * jWeights[3]
                                  * iWeights[2]
                            + srcPtr[(si + 2) * srcRowPitch + (sj + 2) * elementsPerPixel + k] * jWeights[3]
                                  * iWeights[3]));

                        dstPtr[di * dstRowPitch + dj * elementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                    }
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpResize, test::ValueList<int, int, int, int, NVCVInterpolationType, int>
{
    // srcWidth, srcHeight, dstWidth, dstHeight,       interpolation, numberImages
    {        42,        48,       23,        24, NVCV_INTERP_NEAREST,           1},
    {       113,        12,      212,        36, NVCV_INTERP_NEAREST,           1},
    {       421,       148,      223,       124, NVCV_INTERP_NEAREST,           2},
    {       313,       212,      412,       336, NVCV_INTERP_NEAREST,           3},
    {        42,        40,       21,        20,  NVCV_INTERP_LINEAR,           1},
    {        21,        21,       42,        42,  NVCV_INTERP_LINEAR,           1},
    {       420,       420,      210,       210,  NVCV_INTERP_LINEAR,           4},
    {       210,       210,      420,       420,  NVCV_INTERP_LINEAR,           5},
    {        42,        40,       21,        20,   NVCV_INTERP_CUBIC,           1},
    {        21,        21,       42,        42,   NVCV_INTERP_CUBIC,           6},
});

// clang-format on

TEST_P(OpResize, correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();
    int dstWidth  = GetParamValue<2>();
    int dstHeight = GetParamValue<3>();

    NVCVInterpolationType interpolation = GetParamValue<4>();

    int numberOfImages = GetParamValue<5>();

    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, nvcv::FMT_RGBA8);
    nvcv::Tensor imgDst(numberOfImages, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());

    ASSERT_NE(nullptr, srcData);
    ASSERT_NE(nullptr, dstData);

    int srcBufSize = (srcData->imgPitchBytes() / sizeof(uint8_t)) * srcData->numImages();
    int dstBufSize = (dstData->imgPitchBytes() / sizeof(uint8_t)) * dstData->numImages();

    std::vector<uint8_t> srcVec(srcBufSize);

    std::vector<uint8_t> testVec(dstBufSize);
    std::vector<uint8_t> goldVec(dstBufSize);

    test::FillRandomData(srcVec);

    // Copy input data to the GPU
    EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->mem(), srcVec.data(), srcVec.size() * sizeof(uint8_t),
                                           cudaMemcpyHostToDevice, stream));

    // Generate gold result
    Resize(goldVec, srcVec, dstData, srcData, interpolation);

    // Generate test result
    nv::cvop::Resize resizeOp;

    EXPECT_NO_THROW(resizeOp(stream, imgSrc, imgDst, interpolation));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), dstData->mem(), dstBufSize, cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    test::DebugPrintImage(srcVec, srcData->rowPitchBytes() / sizeof(uint8_t));
    test::DebugPrintImage(testVec, dstData->rowPitchBytes() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstData->rowPitchBytes() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec, dstData->rowPitchBytes() / sizeof(uint8_t),
                             dstData->rowPitchBytes() / sizeof(uint8_t));
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}
