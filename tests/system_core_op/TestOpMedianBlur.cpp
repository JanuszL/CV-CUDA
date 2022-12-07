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
#include <operators/OpMedianBlur.hpp>

#include <cmath>
#include <random>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;
namespace t    = ::testing;

// #define DBG_MEDIAN_BLUR 1

static void printVec(std::vector<uint8_t> &vec, int height, int rowPitch, int bytesPerPixel, std::string name)
{
#if DBG_MEDIAN_BLUR
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

static uint8_t computeMedianInSubsetMatrix(const std::vector<uint8_t> &hSrc, int srcRowPitch, nvcv::Size2D srcSize,
                                           nv::cv::Size2D ksize, nvcv::ImageFormat fmt, int x, int y, int z)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    const uint8_t       *srcPtr       = hSrc.data();
    int                  offsetWidth  = ksize.w / 2;
    int                  offsetHeight = ksize.h / 2;
    std::vector<uint8_t> samples;

    for (int j = -offsetHeight; j <= offsetHeight; j++)
    {
        for (int i = -offsetWidth; i <= offsetWidth; i++)
        {
            int yP = y + j;
            int xP = x + i;
            samples.push_back(srcPtr[yP * srcRowPitch + xP * elementsPerPixel + z]);
        }
    }

    EXPECT_EQ(samples.size(), ksize.w * ksize.h);

#if DBG_MEDIAN_BLUR
    std::cout << offsetWidth << " " << offsetHeight << " " << samples.size() << std::endl;

    std::cout << "coord (" << x << ","
              << ") ";
    for (auto &e : samples)
    {
        std::cout << static_cast<int>(e) << "->";
    }
    std::cout << std::endl;
#endif

    sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void GenerateMedianBlurGoldenOutput(std::vector<uint8_t> &hDst, int dstRowPitch, nvcv::Size2D dstSize,
                                           const std::vector<uint8_t> &hSrc, int srcRowPitch, nvcv::Size2D srcSize,
                                           nvcv::ImageFormat fmt, nv::cv::Size2D ksize)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    uint8_t *dstPtr = hDst.data();

    int width  = dstSize.w;
    int height = dstSize.h;

    int offsetWidth  = ksize.w / 2;
    int offsetHeight = ksize.h / 2;

    for (int dst_y = 0; dst_y < height; dst_y++)
    {
        for (int dst_x = 0; dst_x < width; dst_x++)
        {
            for (int k = 0; k < elementsPerPixel; k++)
            {
                dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k] = computeMedianInSubsetMatrix(
                    hSrc, srcRowPitch, srcSize, ksize, fmt, dst_x + offsetWidth, dst_y + offsetHeight, k);
            }
        }
    }
}

static void GenerateInputWithBorderReplicate(std::vector<uint8_t> &hDst, int dstRowPitch, nvcv::Size2D dstSize,
                                             std::vector<uint8_t> &hSrc, int srcRowPitch, nvcv::Size2D srcSize,
                                             nvcv::ImageFormat fmt, nv::cv::Size2D ksize)
{
    assert(fmt.numPlanes() == 1);

    int elementsPerPixel = fmt.numChannels();

    uint8_t *srcPtr = hSrc.data();
    uint8_t *dstPtr = hDst.data();

    int srcWidth  = srcSize.w;
    int srcHeight = srcSize.h;

    int dstWidth  = dstSize.w;
    int dstHeight = dstSize.h;

    int offsetWidth  = ksize.w / 2;
    int offsetHeight = ksize.h / 2;

    for (int dst_y = 0; dst_y < dstHeight; dst_y++)
    {
        for (int dst_x = 0; dst_x < dstWidth; dst_x++)
        {
            for (int k = 0; k < elementsPerPixel; k++)
            {
                int reducedOffsetX = dst_x - offsetWidth;
                int reducedOffsetY = dst_y - offsetHeight;
                if (reducedOffsetX <= 0)
                {
                    reducedOffsetX = 0;
                }
                else if (reducedOffsetX >= srcWidth)
                {
                    reducedOffsetX = srcWidth - 1;
                }

                if (reducedOffsetY <= 0)
                {
                    reducedOffsetY = 0;
                }
                else if (reducedOffsetY >= srcHeight)
                {
                    reducedOffsetY = srcHeight - 1;
                }
                dstPtr[dst_y * dstRowPitch + dst_x * elementsPerPixel + k]
                    = srcPtr[reducedOffsetY * srcRowPitch + reducedOffsetX * elementsPerPixel + k];
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpMedianBlur, test::ValueList<int, int, nv::cv::Size2D, int>
{
    // width,       height,  kernel size, numberImages
    {         9,         9,        {5,5},           1},
    {         9,         9,        {5,5},           4},

    {        21,        21,      {15,15},           1},
    {        21,        21,      {15,15},           4},
});

// clang-format on

TEST_P(OpMedianBlur, tensor_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth  = GetParamValue<0>();
    int srcHeight = GetParamValue<1>();

    nv::cv::Size2D ksize          = GetParamValue<2>();
    int            numberOfImages = GetParamValue<3>();

    int srcBrdReplicateWidth  = srcWidth + (ksize.w / 2) * 2;
    int srcBrdReplicateHeight = srcHeight + (ksize.h / 2) * 2;

    const nvcv::ImageFormat fmt           = nvcv::FMT_RGB8;
    const int               bytesPerPixel = 3;

    // Generate input
    nvcv::Tensor imgSrc(numberOfImages, {srcWidth, srcHeight}, fmt);

    const auto *srcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());

    ASSERT_NE(nullptr, srcData);

    auto srcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcAccess);

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    int                               srcVecRowPitch = srcWidth * fmt.planePixelStrideBytes(0);

    std::vector<std::vector<uint8_t>> srcBrdReplicateVec(numberOfImages);
    int                               srcBrdReplicateVecRowPitch = srcBrdReplicateWidth * fmt.planePixelStrideBytes(0);

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

        printVec(srcVec[i], srcHeight, srcVecRowPitch, bytesPerPixel, "input");

        srcBrdReplicateVec[i].resize(srcBrdReplicateHeight * srcBrdReplicateVecRowPitch);

        GenerateInputWithBorderReplicate(srcBrdReplicateVec[i], srcBrdReplicateVecRowPitch,
                                         {srcBrdReplicateWidth, srcBrdReplicateHeight}, srcVec[i], srcVecRowPitch,
                                         {srcWidth, srcHeight}, fmt, ksize);

        printVec(srcBrdReplicateVec[i], srcBrdReplicateHeight, srcBrdReplicateVecRowPitch, bytesPerPixel,
                 "input with replicated border");
    }

    // Generate test result
    nvcv::Tensor imgDst(numberOfImages, {srcWidth, srcHeight}, fmt);

    nv::cvop::MedianBlur medianBlurOp(0);
    EXPECT_NO_THROW(medianBlurOp(stream, imgSrc, imgDst, ksize));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check result
    const auto *dstData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);

    auto dstAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstAccess);

    int dstVecRowPitch = srcWidth * fmt.planePixelStrideBytes(0);
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        std::vector<uint8_t> testVec(srcHeight * dstVecRowPitch);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstVecRowPitch, dstAccess->sampleData(i), dstAccess->rowPitchBytes(),
                               dstVecRowPitch, // vec has no padding
                               srcHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(srcHeight * dstVecRowPitch);

        // Generate gold result and compare against operator output
        // Note - this function ignores the border pixels and only tests the inner pixels
        GenerateMedianBlurGoldenOutput(goldVec, dstVecRowPitch, {srcWidth, srcHeight}, srcBrdReplicateVec[i],
                                       srcBrdReplicateVecRowPitch, {srcBrdReplicateWidth, srcBrdReplicateHeight}, fmt,
                                       ksize);

        printVec(goldVec, srcHeight, dstVecRowPitch, bytesPerPixel, "golden output");

        printVec(testVec, srcHeight, dstVecRowPitch, bytesPerPixel, "operator output");

        EXPECT_EQ(goldVec, testVec);
    }
}

TEST_P(OpMedianBlur, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidthBase  = GetParamValue<0>();
    int srcHeightBase = GetParamValue<1>();

    nv::cv::Size2D ksize          = GetParamValue<2>();
    int            numberOfImages = GetParamValue<3>();

    const nvcv::ImageFormat fmt           = nvcv::FMT_RGB8;
    const int               bytesPerPixel = 3;

    // Create tensor to store kernel size
    nvcv::Tensor ksizeTensor(nvcv::TensorShape({numberOfImages, 2}, nvcv::TensorLayout::NW), nvcv::TYPE_S32);
    const auto  *ksizeTensorData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(ksizeTensor.exportData());
    ASSERT_NE(nullptr, ksizeTensorData);

    auto ksizeTensorDataAccess = nvcv::TensorDataAccessPitch::Create(*ksizeTensorData);
    ASSERT_TRUE(ksizeTensorDataAccess);

    // Create input and output
    std::default_random_engine         randEng;
    std::uniform_int_distribution<int> rndSrcWidth(srcWidthBase * 0.8, srcWidthBase * 1.1);
    std::uniform_int_distribution<int> rndSrcHeight(srcHeightBase * 0.8, srcHeightBase * 1.1);
    std::uniform_int_distribution<int> rndSrcKSizeWidth(ksize.w * 0.8, ksize.w * 1.1);
    std::uniform_int_distribution<int> rndSrcKSizeHeight(ksize.h * 0.8, ksize.h * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc, imgSrcBrdReplicate, imgDst;
    std::vector<nv::cv::Size2D>               ksizeVecs;

    for (int i = 0; i < numberOfImages; ++i)
    {
        int tmpWidth            = i == 0 ? srcWidthBase : rndSrcWidth(randEng);
        int tmpHeight           = i == 0 ? srcHeightBase : rndSrcHeight(randEng);
        int tmpKernelSizeWidth  = i == 0 ? ksize.w : rndSrcKSizeWidth(randEng);
        int tmpKernelSizeHeight = i == 0 ? ksize.h : rndSrcKSizeHeight(randEng);

        // Width and height needs to be Odd
        tmpKernelSizeWidth  = tmpKernelSizeWidth % 2 == 1 ? tmpKernelSizeWidth : tmpKernelSizeWidth - 1;
        tmpKernelSizeHeight = tmpKernelSizeHeight % 2 == 1 ? tmpKernelSizeHeight : tmpKernelSizeHeight - 1;

        int tmpBrdReplicateWidth  = tmpWidth + (ksize.w / 2) * 2;
        int tmpBrdReplicateHeight = tmpHeight + (ksize.h / 2) * 2;

        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{tmpWidth, tmpHeight}, fmt));
        imgSrcBrdReplicate.emplace_back(
            std::make_unique<nvcv::Image>(nvcv::Size2D{tmpBrdReplicateWidth, tmpBrdReplicateHeight}, fmt));

        imgDst.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{tmpWidth, tmpHeight}, fmt));

        ksizeVecs.push_back({tmpKernelSizeWidth, tmpKernelSizeHeight});
    }

    // Copy the kernel sizes to device tensor
    ASSERT_EQ(cudaSuccess, cudaMemcpy2DAsync(ksizeTensorDataAccess->sampleData(0),
                                             ksizeTensorDataAccess->samplePitchBytes(), ksizeVecs.data(), sizeof(int2),
                                             sizeof(int2), numberOfImages, cudaMemcpyHostToDevice, stream));

    nvcv::ImageBatchVarShape batchSrc(numberOfImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    nvcv::ImageBatchVarShape batchDst(numberOfImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    std::vector<std::vector<uint8_t>> srcVec(numberOfImages);
    std::vector<int>                  srcVecRowPitch(numberOfImages);

    std::vector<std::vector<uint8_t>> srcBrdReplicateVec(numberOfImages);
    std::vector<int>                  srcBrdReplicateVecRowPitch(numberOfImages);

    // Populate input
    for (int i = 0; i < numberOfImages; ++i)
    {
        const auto *srcData = dynamic_cast<const nvcv::IImageDataPitchDevice *>(imgSrc[i]->exportData());
        ASSERT_NE(nullptr, srcData);

        assert(srcData->numPlanes() == 1);

        int srcWidth  = srcData->plane(0).width;
        int srcHeight = srcData->plane(0).height;

        srcVecRowPitch[i] = srcWidth * fmt.planePixelStrideBytes(0);

        std::uniform_int_distribution<uint8_t> rand(0, 255);

        srcVec[i].resize(srcHeight * srcVecRowPitch[i]);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });

        printVec(srcVec[i], srcHeight, srcVecRowPitch[i], bytesPerPixel, "input");

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(srcData->plane(0).buffer, srcData->plane(0).pitchBytes, srcVec[i].data(),
                                            srcVecRowPitch[i],
                                            srcVecRowPitch[i], // vec has no padding
                                            srcHeight, cudaMemcpyHostToDevice));

        // Fill the border with BORDER_REPLICATE
        const auto *srcBrdReplicateData
            = dynamic_cast<const nvcv::IImageDataPitchDevice *>(imgSrcBrdReplicate[i]->exportData());
        ASSERT_NE(nullptr, srcBrdReplicateData);

        assert(srcBrdReplicateData->numPlanes() == 1);

        int srcBrdReplicateWidth  = srcBrdReplicateData->plane(0).width;
        int srcBrdReplicateHeight = srcBrdReplicateData->plane(0).height;

        srcBrdReplicateVecRowPitch[i] = srcBrdReplicateWidth * fmt.planePixelStrideBytes(0);

        srcBrdReplicateVec[i].resize(srcBrdReplicateHeight * srcBrdReplicateVecRowPitch[i]);

        GenerateInputWithBorderReplicate(srcBrdReplicateVec[i], srcBrdReplicateVecRowPitch[i],
                                         {srcBrdReplicateWidth, srcBrdReplicateHeight}, srcVec[i], srcVecRowPitch[i],
                                         {srcWidth, srcHeight}, fmt, ksizeVecs[i]);

        printVec(srcBrdReplicateVec[i], srcBrdReplicateHeight, srcBrdReplicateVecRowPitch[i], bytesPerPixel,
                 "input with replicated border");
    }

    // Generate test result
    nv::cvop::MedianBlur medianBlurOp(numberOfImages);
    EXPECT_NO_THROW(medianBlurOp(stream, batchSrc, batchDst, ksizeTensor));

    // Get test data back
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numberOfImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcBrdReplicateData
            = dynamic_cast<const nvcv::IImageDataPitchDevice *>(imgSrcBrdReplicate[i]->exportData());
        assert(srcBrdReplicateData->numPlanes() == 1);
        int srcBrdReplicateWidth  = srcBrdReplicateData->plane(0).width;
        int srcBrdReplicateHeight = srcBrdReplicateData->plane(0).height;

        const auto *dstData = dynamic_cast<const nvcv::IImageDataPitchDevice *>(imgDst[i]->exportData());
        assert(dstData->numPlanes() == 1);

        int dstWidth  = dstData->plane(0).width;
        int dstHeight = dstData->plane(0).height;

        int dstRowPitch                = dstWidth * fmt.planePixelStrideBytes(0);
        int srcBrdReplicateVecRowPitch = srcBrdReplicateWidth * fmt.planePixelStrideBytes(0);

        std::vector<uint8_t> testVec(dstHeight * dstRowPitch);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowPitch, dstData->plane(0).buffer, dstData->plane(0).pitchBytes,
                               dstRowPitch, // vec has no padding
                               dstHeight, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(dstHeight * dstRowPitch);
        std::generate(goldVec.begin(), goldVec.end(), [&]() { return 0; });

        // Generate gold result
        GenerateMedianBlurGoldenOutput(goldVec, dstRowPitch, {dstWidth, dstHeight}, srcBrdReplicateVec[i],
                                       srcBrdReplicateVecRowPitch, {srcBrdReplicateWidth, srcBrdReplicateHeight}, fmt,
                                       ksizeVecs[i]);

        printVec(goldVec, dstHeight, dstRowPitch, bytesPerPixel, "golden output");

        printVec(testVec, dstHeight, dstRowPitch, bytesPerPixel, "operator output");

        EXPECT_EQ(goldVec, testVec);
    }
}
