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
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpNormalize.hpp>

//#define DEBUG_PRINT_IMAGE
//#define DEBUG_PRINT_DIFF

#include <common/Utils.hpp>

#include <cmath>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;

static void Normalize(std::vector<uint8_t> &hDst, const std::vector<uint8_t> &hSrc, const std::vector<float> &hBase,
                      const std::vector<float> &hScale, const nvcv::TensorDataAccessPitchImagePlanar &dSrcData,
                      const nvcv::TensorDataAccessPitchImagePlanar &dBaseData,
                      const nvcv::TensorDataAccessPitchImagePlanar &dScaleData, const float globalScale,
                      const float globalShift, const float epsilon, const uint32_t flags)
{
    int inoutImgPitch         = dSrcData.samplePitchBytes() / sizeof(uint8_t);
    int inoutRowPitch         = dSrcData.rowPitchBytes() / sizeof(uint8_t);
    int inoutNumImages        = dSrcData.numSamples();
    int inoutWidth            = dSrcData.numCols();
    int inoutHeight           = dSrcData.numRows();
    int inoutChannels         = dSrcData.numChannels();
    int inoutElementsPerPixel = dSrcData.numChannels();
    int baseImgPitch          = dBaseData.samplePitchBytes() / sizeof(float);
    int scaleImgPitch         = dScaleData.samplePitchBytes() / sizeof(float);
    int baseRowPitch          = dBaseData.rowPitchBytes() / sizeof(float);
    int scaleRowPitch         = dScaleData.rowPitchBytes() / sizeof(float);
    int baseElementsPerPixel  = dBaseData.numChannels();
    int scaleElementsPerPixel = dScaleData.numChannels();

    uint8_t       *dstPtrTop   = hDst.data();
    const uint8_t *srcPtrTop   = hSrc.data();
    const float   *basePtrTop  = hBase.data();
    const float   *scalePtrTop = hScale.data();

    using FT = float;

    for (int img = 0; img < inoutNumImages; img++)
    {
        const int bimg = (dBaseData.numSamples() == 1 ? 0 : img);
        const int simg = (dScaleData.numSamples() == 1 ? 0 : img);

        uint8_t       *dstPtr   = dstPtrTop + inoutImgPitch * img;
        const uint8_t *srcPtr   = srcPtrTop + inoutImgPitch * img;
        const float   *basePtr  = basePtrTop + baseImgPitch * bimg;
        const float   *scalePtr = scalePtrTop + scaleImgPitch * simg;

        for (int i = 0; i < inoutHeight; i++)
        {
            const int bi = (dBaseData.numRows() == 1 ? 0 : i);
            const int si = (dScaleData.numRows() == 1 ? 0 : i);

            for (int j = 0; j < inoutWidth; j++)
            {
                const int bj = (dBaseData.numCols() == 1 ? 0 : j);
                const int sj = (dScaleData.numCols() == 1 ? 0 : j);

                for (int k = 0; k < inoutChannels; k++)
                {
                    const int bk = (dBaseData.numChannels() == 1 ? 0 : k);
                    const int sk = (dScaleData.numChannels() == 1 ? 0 : k);

                    FT mul;

                    if (flags & NVCV_OP_NORMALIZE_SCALE_IS_STDDEV)
                    {
                        FT s = scalePtr[si * scaleRowPitch + sj * scaleElementsPerPixel + sk];
                        FT x = s * s + epsilon;
                        mul  = FT{1} / std::sqrt(x);
                    }
                    else
                    {
                        mul = scalePtr[si * scaleRowPitch + sj * scaleElementsPerPixel + sk];
                    }

                    FT res = std::rint((srcPtr[i * inoutRowPitch + j * inoutElementsPerPixel + k]
                                        - basePtr[bi * baseRowPitch + bj * baseElementsPerPixel + bk])
                                           * mul * globalScale
                                       + globalShift);

                    dstPtr[i * inoutRowPitch + j * inoutElementsPerPixel + k] = res < 0 ? 0 : (res > 255 ? 255 : res);
                }
            }
        }
    }
}

static uint32_t normalScale   = 0;
static uint32_t scaleIsStdDev = NVCV_OP_NORMALIZE_SCALE_IS_STDDEV;

// clang-format off

NVCV_TEST_SUITE_P(OpNormalize, test::ValueList<int, int, int, bool, bool, uint32_t, float, float, float>
{
    // width, height, numImages, scalarBase, scalarScale,         flags, globalScale, globalShift, epsilon,
    {     32,     33,         1,       true,        true,   normalScale,         0.f,         0.f,     0.f, },
    {     66,     55,         1,       true,        true,   normalScale,         1.f,         0.f,     0.f, },
    {    122,    212,         2,       true,        true,   normalScale,      1.234f,      43.21f,     0.f, },
    {    211,    102,         3,      false,       false,   normalScale,        1.1f,        0.1f,     0.f, },
    {     21,     12,         5,       true,        true, scaleIsStdDev,        1.2f,        0.2f,     0.f, },
    {     63,     32,         7,      false,        true,   normalScale,        1.3f,        0.3f,     0.f, },
    {     22,     13,         9,       true,       false,   normalScale,        1.4f,        0.4f,     0.f, },
    {     55,     33,         2,       true,       false, scaleIsStdDev,        2.1f,        1.1f,   1.23f, },
    {    444,    222,         4,       true,       false, scaleIsStdDev,        2.2f,        2.2f,   12.3f, },
});

// clang-format on

TEST_P(OpNormalize, correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int      width       = GetParamValue<0>();
    int      height      = GetParamValue<1>();
    int      numImages   = GetParamValue<2>();
    bool     scalarBase  = GetParamValue<3>();
    bool     scalarScale = GetParamValue<4>();
    uint32_t flags       = GetParamValue<5>();
    float    globalScale = GetParamValue<6>();
    float    globalShift = GetParamValue<7>();
    float    epsilon     = GetParamValue<8>();

    int baseWidth      = (scalarBase ? 1 : width);
    int scaleWidth     = (scalarScale ? 1 : width);
    int baseHeight     = (scalarBase ? 1 : height);
    int scaleHeight    = (scalarScale ? 1 : height);
    int baseNumImages  = (scalarBase ? 1 : numImages);
    int scaleNumImages = (scalarScale ? 1 : numImages);

    nvcv::ImageFormat baseFormat  = (scalarBase ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);
    nvcv::ImageFormat scaleFormat = (scalarScale ? nvcv::FMT_F32 : nvcv::FMT_RGBAf32);

    nvcv::Tensor imgSrc(numImages, {width, height}, nvcv::FMT_RGBA8);
    nvcv::Tensor imgDst(numImages, {width, height}, nvcv::FMT_RGBA8);
    nvcv::Tensor imgBase(baseNumImages, {baseWidth, baseHeight}, baseFormat);
    nvcv::Tensor imgScale(scaleNumImages, {scaleWidth, scaleHeight}, scaleFormat);

    const auto *srcData   = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());
    const auto *dstData   = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());
    const auto *baseData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgBase.exportData());
    const auto *scaleData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgScale.exportData());

    ASSERT_NE(nullptr, srcData);
    ASSERT_NE(nullptr, dstData);
    ASSERT_NE(nullptr, baseData);
    ASSERT_NE(nullptr, scaleData);

    auto srcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    auto dstAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    auto baseAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*baseData);
    ASSERT_TRUE(baseAccess);

    auto scaleAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*scaleData);
    ASSERT_TRUE(scaleAccess);

    int inoutBufSize = (srcAccess->samplePitchBytes() / sizeof(uint8_t)) * srcAccess->numSamples();
    int baseBufSize  = (baseAccess->samplePitchBytes() / sizeof(float)) * baseAccess->numSamples();
    int scaleBufSize = (scaleAccess->samplePitchBytes() / sizeof(float)) * scaleAccess->numSamples();

    std::vector<uint8_t> srcVec(inoutBufSize);
    std::vector<uint8_t> testVec(inoutBufSize);
    std::vector<uint8_t> goldVec(inoutBufSize);
    std::vector<float>   baseVec(baseBufSize);
    std::vector<float>   scaleVec(scaleBufSize);

    test::FillRandomData(srcVec);
    test::FillRandomData(baseVec, 0.f, 255.f);
    test::FillRandomData(scaleVec, 0.f, 1.f);

    // Copy input data to the GPU
    EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(srcData->data(), srcVec.data(), srcVec.size() * sizeof(uint8_t),
                                           cudaMemcpyHostToDevice, stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(baseData->data(), baseVec.data(), baseVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpyAsync(scaleData->data(), scaleVec.data(), scaleVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    // Generate gold result
    Normalize(goldVec, srcVec, baseVec, scaleVec, *srcAccess, *baseAccess, *scaleAccess, globalScale, globalShift,
              epsilon, flags);

    // Generate test result
    nv::cvop::Normalize normalizeOp;

    EXPECT_NO_THROW(normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, globalScale, globalShift, epsilon, flags));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), dstData->data(), inoutBufSize, cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    test::DebugPrintImage(srcVec, srcAccess->rowPitchBytes() / sizeof(uint8_t));
    test::DebugPrintImage(baseVec, baseAccess->rowPitchBytes() / sizeof(float));
    test::DebugPrintImage(scaleVec, scaleAccess->rowPitchBytes() / sizeof(float));
    test::DebugPrintImage(testVec, dstAccess->rowPitchBytes() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstAccess->rowPitchBytes() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec, dstAccess->rowPitchBytes() / sizeof(uint8_t),
                             dstAccess->rowPitchBytes() / sizeof(uint8_t));
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}
