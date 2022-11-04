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

#include <common/Utils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpNormalize.hpp>

#include <cmath>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;

static void Normalize(std::vector<uint8_t> &hDst, int dstRowPitch, const std::vector<uint8_t> &hSrc, int srcRowPitch,
                      nvcv::Size2D size, nvcv::ImageFormat fmt, const std::vector<float> &hBase, int baseRowPitch,
                      nvcv::Size2D baseSize, nvcv::ImageFormat baseFormat, const std::vector<float> &hScale,
                      int scaleRowPitch, nvcv::Size2D scaleSize, nvcv::ImageFormat scaleFormat, const float globalScale,
                      const float globalShift, const float epsilon, const uint32_t flags)
{
    using FT = float;

    for (int i = 0; i < size.h; i++)
    {
        const int bi = baseSize.h == 1 ? 0 : i;
        const int si = scaleSize.h == 1 ? 0 : i;

        for (int j = 0; j < size.w; j++)
        {
            const int bj = baseSize.w == 1 ? 0 : j;
            const int sj = scaleSize.w == 1 ? 0 : j;

            for (int k = 0; k < fmt.numChannels(); k++)
            {
                const int bk = (baseFormat.numChannels() == 1 ? 0 : k);
                const int sk = (scaleFormat.numChannels() == 1 ? 0 : k);

                FT mul;

                if (flags & NVCV_OP_NORMALIZE_SCALE_IS_STDDEV)
                {
                    FT s = hScale.at(si * scaleRowPitch + sj * scaleFormat.numChannels() + sk);
                    FT x = s * s + epsilon;
                    mul  = FT{1} / std::sqrt(x);
                }
                else
                {
                    mul = hScale.at(si * scaleRowPitch + sj * scaleFormat.numChannels() + sk);
                }

                FT res = std::rint((hSrc.at(i * srcRowPitch + j * fmt.numChannels() + k)
                                    - hBase.at(bi * baseRowPitch + bj * baseFormat.numChannels() + bk))
                                       * mul * globalScale
                                   + globalShift);

                hDst.at(i * dstRowPitch + j * fmt.numChannels() + k) = res < 0 ? 0 : (res > 255 ? 255 : res);
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

    nvcv::ImageFormat fmt = nvcv::FMT_RGBA8;

    std::default_random_engine rng;

    // Create input tensor
    nvcv::Tensor imgSrc(numImages, {width, height}, fmt);
    const auto  *srcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    int                               srcVecRowPitch = width * fmt.numChannels();
    for (int i = 0; i < numImages; ++i)
    {
        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(height * srcVecRowPitch);
        generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(srcAccess->sampleData(i), srcAccess->rowPitchBytes(), srcVec[i].data(), srcVecRowPitch,
                               srcVecRowPitch, // vec has no padding
                               height, cudaMemcpyHostToDevice));
    }

    // Create base tensor
    nvcv::Tensor imgBase(baseNumImages, {baseWidth, baseHeight}, baseFormat);
    const auto  *baseData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgBase.exportData());
    ASSERT_NE(nullptr, baseData);
    auto baseAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*baseData);
    ASSERT_TRUE(baseAccess);

    std::vector<std::vector<float>> baseVec(baseNumImages);
    int                             baseVecRowPitch = baseWidth * baseFormat.numChannels();
    for (int i = 0; i < baseNumImages; ++i)
    {
        std::uniform_real_distribution<float> udist(0, 255.f);

        baseVec[i].resize(baseHeight * baseVecRowPitch);
        generate(baseVec[i].begin(), baseVec[i].end(), [&]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(baseAccess->sampleData(i), baseAccess->rowPitchBytes(), baseVec[i].data(),
                                            baseVecRowPitch * sizeof(float),
                                            baseVecRowPitch * sizeof(float), // vec has no padding
                                            baseHeight, cudaMemcpyHostToDevice));
    }

    // Create scale tensor
    nvcv::Tensor imgScale(scaleNumImages, {scaleWidth, scaleHeight}, scaleFormat);
    const auto  *scaleData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgScale.exportData());
    ASSERT_NE(nullptr, scaleData);
    auto scaleAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*scaleData);
    ASSERT_TRUE(scaleAccess);

    std::vector<std::vector<float>> scaleVec(scaleNumImages);
    assert(scaleFormat.numPlanes() == 1);
    int scaleVecRowPitch = scaleWidth * scaleFormat.numChannels();
    for (int i = 0; i < scaleNumImages; ++i)
    {
        std::uniform_real_distribution<float> udist(0, 1.f);

        scaleVec[i].resize(scaleHeight * scaleVecRowPitch);
        generate(scaleVec[i].begin(), scaleVec[i].end(), [&]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(scaleAccess->sampleData(i), scaleAccess->rowPitchBytes(),
                                            scaleVec[i].data(), scaleVecRowPitch * sizeof(float),
                                            scaleVecRowPitch * sizeof(float), // vec has no padding
                                            scaleHeight, cudaMemcpyHostToDevice));
    }

    // Create dest tensor
    nvcv::Tensor imgDst(numImages, {width, height}, nvcv::FMT_RGBA8);

    // Generate test result
    nv::cvop::Normalize normalizeOp;
    EXPECT_NO_THROW(normalizeOp(stream, imgSrc, imgBase, imgScale, imgDst, globalScale, globalShift, epsilon, flags));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowPitch = width * fmt.numChannels();
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(height * dstVecRowPitch);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowPitch, dstAccess->sampleData(i), dstAccess->rowPitchBytes(),
                               dstVecRowPitch, // vec has no padding
                               height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(height * dstVecRowPitch);

        int bi = baseNumImages == 1 ? 0 : i;
        int si = scaleNumImages == 1 ? 0 : i;

        // Generate gold result
        Normalize(goldVec, dstVecRowPitch, srcVec[i], srcVecRowPitch, {width, height}, fmt, baseVec[bi],
                  baseVecRowPitch, {baseWidth, baseHeight}, baseFormat, scaleVec[si], scaleVecRowPitch,
                  {scaleWidth, scaleHeight}, scaleFormat, globalScale, globalShift, epsilon, flags);

        EXPECT_EQ(goldVec, testVec);
    }
}
