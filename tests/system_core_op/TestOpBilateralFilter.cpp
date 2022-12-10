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
#include <operators/OpBilateralFilter.hpp>

#include <iostream>
#include <random>
#include <vector>

namespace nvcv = nv::cv;
namespace gt   = ::testing;
namespace test = nv::cv::test;

static uint32_t saturate_cast(float n)
{
    return static_cast<uint32_t>(std::min(255.0f, std::round(n)));
}

static bool CompareTensors(std::vector<uint8_t> &vTest, std::vector<uint8_t> &vGold, size_t columns, size_t rows,
                           size_t batch, size_t rowPitch, size_t samplePitch, float delta)
{
    for (size_t i = 0; i < batch; i++)
    {
        uint8_t *pTest = vTest.data() + i * samplePitch;
        uint8_t *pGold = vGold.data() + i * samplePitch;
        for (size_t j = 0; j < rows; j++)
        {
            for (size_t k = 0; k < columns; k++)
            {
                size_t offset = j * rowPitch + k;
                float  diff   = std::abs(static_cast<float>(pTest[offset]) - static_cast<float>(pGold[offset]));
                if (diff > delta)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

static void CPUBilateralFilter(std::vector<uint8_t> &vIn, std::vector<uint8_t> &vOut, int columns, int rows, int batch,
                               int rowPitch, int samplePitch, int d, float sigmaColor, float sigmaSpace)
{
    int   radius            = d / 2;
    float radiusSquared     = radius * radius;
    float space_coefficient = -1 / (2 * sigmaSpace * sigmaSpace);
    float color_coefficient = -1 / (2 * sigmaColor * sigmaColor);
    for (int i = 0; i < batch; i++)
    {
        uint8_t *pIn  = vIn.data() + i * samplePitch;
        uint8_t *pOut = vOut.data() + i * samplePitch;
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < columns; k++)
            {
                float numerator   = 0.0f;
                float denominator = 0.0f;
                float center      = static_cast<float>(pIn[j * rowPitch + k]);
                for (int y = j - radius; y <= j + radius; y++)
                {
                    for (int x = k - radius; x <= k + radius; x++)
                    {
                        float distanceSquared = (k - x) * (k - x) + (j - y) * (j - y);
                        if (distanceSquared <= radiusSquared)
                        {
                            float pixel         = ((x >= 0) && (x < columns) && (y >= 0) && (y < rows))
                                                    ? static_cast<float>(pIn[y * rowPitch + x])
                                                    : 0.0f;
                            float e_space       = distanceSquared * space_coefficient;
                            float one_norm_size = std::abs(pixel - center);
                            float e_color       = one_norm_size * one_norm_size * color_coefficient;
                            float weight        = std::exp(e_space + e_color);
                            denominator += weight;
                            numerator += weight * pixel;
                        }
                    }
                }
                pOut[j * rowPitch + k] = saturate_cast(numerator / denominator);
            }
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpBilateralFilter, test::ValueList<int, int, int, float, float, int>
{
    //width, height, d, SigmaColor, sigmaSpace, numberImages
    {    32,     48, 4, 5,          3,          1},
    {    48,     32, 4, 5,          3,          1},
    {    64,     32, 4, 5,          3,          1},
    {    32,    128, 4, 5,          3,          1},

    //width, height, d, SigmaColor, sigmaSpace, numberImages
    {    32,     48, 4, 5,          3,          5},
    {    12,    32,  4, 5,          3,          5},
    {    64,    32,  4, 5,          3,          5},
    {    32,    128, 4, 5,          3,          5},

    //width, height, d, SigmaColor, sigmaSpace, numberImages
    {    32,     48, 4, 5,          3,          9},
    {    48,     32, 4, 5,          3,          9},
    {    64,     32, 4, 5,          3,          9},
    {    32,    128, 4, 5,          3,          9},
});

// clang-format on
TEST_P(OpBilateralFilter, BilateralFilter_packed)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int   width          = GetParamValue<0>();
    int   height         = GetParamValue<1>();
    int   d              = GetParamValue<2>();
    float sigmaColor     = GetParamValue<3>();
    float sigmaSpace     = GetParamValue<4>();
    int   numberOfImages = GetParamValue<5>();

    nvcv::Tensor imgOut(numberOfImages, {width, height}, nvcv::FMT_U8);
    nvcv::Tensor imgIn(numberOfImages, {width, height}, nvcv::FMT_U8);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgIn.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgOut.exportData());

    ASSERT_NE(nullptr, inData);
    ASSERT_NE(nullptr, outData);

    auto inAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);

    auto outAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int inBufSize = inAccess->samplePitchBytes()
                  * inAccess->numSamples(); //img pitch bytes can be more than the image 64, 128, etc
    int outBufSize = outAccess->samplePitchBytes() * outAccess->numSamples();

    std::vector<uint8_t> vIn(inBufSize);
    std::vector<uint8_t> vOut(outBufSize);

    std::vector<uint8_t> inGold(inBufSize, 0);
    std::vector<uint8_t> outGold(outBufSize, 0);
    for (int i = 0; i < inBufSize; i++) inGold[i] = i % 113; // Use prime number to prevent weird tiling patterns

    EXPECT_EQ(cudaSuccess, cudaMemcpy(inData->data(), inGold.data(), inBufSize, cudaMemcpyHostToDevice));
    CPUBilateralFilter(inGold, outGold, inAccess->numCols(), inAccess->numRows(), inAccess->numSamples(),
                       inAccess->rowPitchBytes(), inAccess->samplePitchBytes(), d, sigmaColor, sigmaSpace);

    // run operator
    nv::cvop::BilateralFilter bilateralFilterOp;

    EXPECT_NO_THROW(bilateralFilterOp(stream, imgIn, imgOut, d, sigmaColor, sigmaSpace, NVCV_BORDER_CONSTANT));

    // check cdata
    std::vector<uint8_t> outTest(outBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(outTest.data(), outData->data(), outBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_TRUE(CompareTensors(outTest, outGold, inAccess->numCols(), inAccess->numRows(), inAccess->numSamples(),
                               inAccess->rowPitchBytes(), inAccess->samplePitchBytes(), 0.9f));
}
