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
#include <operators/OpComposite.hpp>

#include <iostream>
#include <random>

namespace nvcv = nv::cv;
namespace gt   = ::testing;
namespace test = nv::cv::test;

//#define DBG_COMPOSITE 1

template<typename T>
static void print_img(std::vector<T> vec, int rowPitch, int height, std::string message)
{
    std::cout << std::endl;
    std::cout << message << std::endl;
    for (int k = 0; k < height; k++)
    {
        for (int j = 0; j < rowPitch; j++)
        {
            std::cout << static_cast<int>(vec[k * rowPitch + j]) << ",";
        }
        std::cout << std::endl;
    }
}

static void setGoldBuffer(std::vector<uint8_t> &gold, std::vector<uint8_t> &fg, std::vector<uint8_t> bg,
                          std::vector<uint8_t> fgMask, int width, int height, int inVecRowPitch, int fgMaskVecRowPitch,
                          int outVecRowPitch, int inChannels, int outChannels)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int      fg_offset     = r * inVecRowPitch + c * inChannels;
            int      fgMask_offset = r * fgMaskVecRowPitch + c;
            int      dst_offset    = r * outVecRowPitch + c * outChannels;
            uint8_t *ptrGold       = gold.data() + dst_offset;
            uint8_t *ptrFg         = fg.data() + fg_offset;
            uint8_t *ptrBg         = bg.data() + fg_offset;
            uint8_t *ptrMat        = fgMask.data() + fgMask_offset;
            uint8_t  a             = *ptrMat;
            for (int k = 0; k < inChannels; k++)
            {
                int c0     = ptrBg[k];
                int c1     = ptrFg[k];
                ptrGold[k] = (uint8_t)(((int)c1 - (int)c0) * (int)a * (1.0f / 255.0f) + c0 + 0.5f);
            }
            if (inChannels == 3 && outChannels == 4)
            {
                ptrGold[3] = 255;
            }
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpComposite, test::ValueList<int, int, int, int, int>
{
    //inWidth, inHeight, in_channels, out_channels, numberImages
    {       5,        4,          3,             3,          1},
    {       5,        4,          3,             3,          4},
    {       5,        4,          3,             4,          1},
    {       5,        4,          3,             4,          4},

    //inWidth, inHeight, in_channels, out_channels, numberImages
    {       4,        4,          3,             3,          1},
    {       4,        4,          3,             3,          4},
    {       4,        4,          3,             4,          1},
    {       4,        4,          3,             4,          4},
});

// clang-format on
TEST_P(OpComposite, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int inWidth        = GetParamValue<0>();
    int inHeight       = GetParamValue<1>();
    int inChannels     = GetParamValue<2>();
    int outChannels    = GetParamValue<3>();
    int numberOfImages = GetParamValue<4>();

    int outWidth  = inWidth;
    int outHeight = inHeight;

    nvcv::ImageFormat inFormat, outFormat;

    if (inChannels == 3)
        inFormat = nvcv::FMT_RGB8;
    if (inChannels == 4)
        inFormat = nvcv::FMT_RGBA8;
    if (outChannels == 3)
        outFormat = nvcv::FMT_RGB8;
    if (outChannels == 4)
        outFormat = nvcv::FMT_RGBA8;

    assert(inChannels <= outChannels);

    nvcv::Tensor foregroundImg(numberOfImages, {inWidth, inHeight}, inFormat);
    nvcv::Tensor backgroundImg(numberOfImages, {inWidth, inHeight}, inFormat);
    nvcv::Tensor fgMaskImg(numberOfImages, {inWidth, inHeight}, nvcv::FMT_U8);
    nvcv::Tensor outImg(numberOfImages, {outWidth, outHeight}, outFormat);

    const auto *foregroundData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(foregroundImg.exportData());
    const auto *backgroundData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(backgroundImg.exportData());
    const auto *fgMaskData     = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(fgMaskImg.exportData());

    ASSERT_NE(nullptr, foregroundData);
    ASSERT_NE(nullptr, backgroundData);
    ASSERT_NE(nullptr, fgMaskData);

    auto foregroundAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*foregroundData);
    ASSERT_TRUE(foregroundAccess);

    auto backgroundAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*backgroundData);
    ASSERT_TRUE(foregroundAccess);

    auto fgMaskAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*fgMaskData);
    ASSERT_TRUE(fgMaskAccess);

    int foregroundBufSize = foregroundAccess->samplePitchBytes()
                          * foregroundAccess->numSamples(); //img pitch bytes can be more than the image 64, 128, etc
    int fgMaskBufSize = fgMaskAccess->samplePitchBytes() * fgMaskAccess->numSamples();

    EXPECT_EQ(cudaSuccess, cudaMemset(foregroundData->data(), 0x00, foregroundBufSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(backgroundData->data(), 0x00, foregroundBufSize));
    EXPECT_EQ(cudaSuccess, cudaMemset(fgMaskData->data(), 0x00, fgMaskBufSize));

    std::vector<std::vector<uint8_t>> foregroundVec(numberOfImages);
    std::vector<std::vector<uint8_t>> backgroundVec(numberOfImages);
    std::vector<std::vector<uint8_t>> fgMaskVec(numberOfImages);

    std::default_random_engine rng;

    int inVecRowPitch     = inWidth * inFormat.planePixelStrideBytes(0);
    int fgMaskVecRowPitch = inWidth * nvcv::FMT_U8.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; i++)
    {
        foregroundVec[i].resize(inHeight * inVecRowPitch);
        backgroundVec[i].resize(inHeight * inVecRowPitch);
        fgMaskVec[i].resize(inHeight * fgMaskVecRowPitch);

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        std::generate(foregroundVec[i].begin(), foregroundVec[i].end(), [&]() { return udist(rng); });
        std::generate(backgroundVec[i].begin(), backgroundVec[i].end(), [&]() { return udist(rng); });
        std::generate(fgMaskVec[i].begin(), fgMaskVec[i].end(), [&]() { return udist(rng); });

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(foregroundAccess->sampleData(i), foregroundAccess->rowPitchBytes(),
                                            foregroundVec[i].data(), inVecRowPitch, inVecRowPitch, inHeight,
                                            cudaMemcpyHostToDevice));
        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(backgroundAccess->sampleData(i), backgroundAccess->rowPitchBytes(),
                                            backgroundVec[i].data(), inVecRowPitch, inVecRowPitch, inHeight,
                                            cudaMemcpyHostToDevice));
        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(fgMaskAccess->sampleData(i), fgMaskAccess->rowPitchBytes(), fgMaskVec[i].data(),
                               fgMaskVecRowPitch, fgMaskVecRowPitch, inHeight, cudaMemcpyHostToDevice));
    }

    // run operator
    nv::cvop::Composite compositeOp;

    EXPECT_NO_THROW(compositeOp(stream, foregroundImg, backgroundImg, fgMaskImg, outImg));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // check cdata
    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(outImg.exportData());
    ASSERT_NE(nullptr, outData);

    auto outAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int outVecRowPitch = outWidth * outFormat.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(inHeight * outVecRowPitch);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), outVecRowPitch, outAccess->sampleData(i), outAccess->rowPitchBytes(),
                               outVecRowPitch, outHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(outHeight * outVecRowPitch);
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

        // generate gold result
        setGoldBuffer(goldVec, foregroundVec[i], backgroundVec[i], fgMaskVec[i], inWidth, inHeight, inVecRowPitch,
                      fgMaskVecRowPitch, outVecRowPitch, inChannels, outChannels);

#ifdef DBG_COMPOSITE
        print_img<uint8_t>(foregroundVec[i], inVecRowPitch, inHeight, "Foreground");
        print_img<uint8_t>(backgroundVec[i], inVecRowPitch, inHeight, "Background");
        print_img<uint8_t>(fgMaskVec[i], fgMaskVecRowPitch, inHeight, "Foreground Mask");
        print_img<uint8_t>(goldVec, outVecRowPitch, outHeight, "Golden output");
        print_img<uint8_t>(testVec, outVecRowPitch, outHeight, "Test output");
#endif
        EXPECT_EQ(goldVec, testVec);
    }
}
