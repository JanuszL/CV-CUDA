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

#include <common/ConvUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpGaussian.hpp>

#include <random>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;
namespace cuda = nv::cv::cuda;

// clang-format off

NVCV_TEST_SUITE_P(OpGaussian, test::ValueList<int, int, int, NVCVImageFormat, int, int, double, double, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, sigmaX, sigmaY,           borderMode
    {    176,    113,       1,      NVCV_IMAGE_FORMAT_U8,      3,      3,    0.5,    0.5, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,      NVCV_IMAGE_FORMAT_U8,      5,      5,   0.75,   0.75, NVCV_BORDER_CONSTANT},
    {    123,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    1.0,    1.0, NVCV_BORDER_WRAP},
    {     42,     53,       4,   NVCV_IMAGE_FORMAT_RGBA8,      7,      7,    0.4,    0.4, NVCV_BORDER_REPLICATE},
    {     13,     42,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,    0.9,    0.9, NVCV_BORDER_REFLECT},
    {     62,    111,       4,   NVCV_IMAGE_FORMAT_RGBA8,      9,      9,    0.8,    0.8, NVCV_BORDER_REFLECT101}
});

// clang-format on

TEST_P(OpGaussian, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    nvcv::Tensor inTensor(batches, {width, height}, format);
    nvcv::Tensor outTensor(batches, {width, height}, format);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(inTensor.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(outTensor.exportData());

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    long3 inPitches{inData->pitchBytes(0), inData->pitchBytes(1), inData->pitchBytes(2)};
    long3 outPitches{outData->pitchBytes(0), outData->pitchBytes(1), outData->pitchBytes(2)};

    long inBufSize  = inPitches.x * inData->shape(0);
    long outBufSize = outPitches.x * outData->shape(0);

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->data(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    nv::cvop::Gaussian gaussianOp(kernelSize, 1);

    EXPECT_NO_THROW(gaussianOp(stream, inTensor, outTensor, kernelSize, sigma, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->data(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    std::vector<float> kernel(kernelSize.w * kernelSize.h);

    int2 half{kernelSize.w / 2, kernelSize.h / 2};

    float sx  = 2.f * sigma.x * sigma.x;
    float sy  = 2.f * sigma.y * sigma.y;
    float s   = 2.f * sigma.x * sigma.y * M_PI;
    float sum = 0.f;
    for (int y = -half.y; y <= half.y; ++y)
    {
        for (int x = -half.x; x <= half.x; ++x)
        {
            float kv = std::exp(-((x * x) / sx + (y * y) / sy)) / s;

            kernel[(y + half.y) * kernelSize.w + (x + half.x)] = kv;

            sum += kv;
        }
    }
    for (int i = 0; i < kernelSize.w * kernelSize.h; ++i)
    {
        kernel[i] /= sum;
    }

    test::Convolve(goldVec, outPitches, inVec, inPitches, shape, format, kernel, kernelSize, kernelAnchor, borderMode,
                   borderValue);

    EXPECT_EQ(testVec, goldVec);
}

TEST_P(OpGaussian, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int    ksizeX = GetParamValue<4>();
    int    ksizeY = GetParamValue<5>();
    double sigmaX = GetParamValue<6>();
    double sigmaY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    double2 sigma{sigmaX, sigmaY};
    int2    kernelAnchor{-1, -1};

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<std::unique_ptr<nv::cv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batches);
    std::vector<int>                  srcVecRowPitch(batches);

    for (int i = 0; i < batches; ++i)
    {
        imgSrc.emplace_back(std::make_unique<nv::cv::Image>(nv::cv::Size2D{udistWidth(rng), udistHeight(rng)}, format));

        int srcRowPitch   = imgSrc[i]->size().w * format.planePixelStrideBytes(0);
        srcVecRowPitch[i] = srcRowPitch;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i]->size().h * srcRowPitch);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto *imgData = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(imgSrc[i]->exportData());
        ASSERT_NE(imgData, nullptr);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).buffer, imgData->plane(0).pitchBytes, srcVec[i].data(),
                                    srcRowPitch, srcRowPitch, imgSrc[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nv::cv::ImageBatchVarShape batchSrc(batches);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<std::unique_ptr<nv::cv::Image>> imgDst;
    for (int i = 0; i < batches; ++i)
    {
        imgDst.emplace_back(std::make_unique<nv::cv::Image>(imgSrc[i]->size(), imgSrc[i]->format()));
    }
    nv::cv::ImageBatchVarShape batchDst(batches);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel size tensor
    nv::cv::Tensor kernelSizeTensor({{batches}, "N"}, nv::cv::TYPE_2S32);
    {
        auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(kernelSizeTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(batches, int2{ksizeX, ksizeY});

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(dev->data(), vec.data(), vec.size() * sizeof(int2), cudaMemcpyHostToDevice, stream));
    }

    // Create sigma tensor
    nv::cv::Tensor sigmaTensor({{batches}, "N"}, nv::cv::TYPE_2F64);
    {
        auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(sigmaTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<double2> vec(batches, sigma);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->data(), vec.data(), vec.size() * sizeof(double2),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    nv::cvop::Gaussian gaussianOp(kernelSize, batches);

    EXPECT_NO_THROW(gaussianOp(stream, batchSrc, batchDst, kernelSizeTensor, sigmaTensor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < batches; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(imgSrc[i]->exportData());
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto *dstData = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(imgDst[i]->exportData());
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowPitch = srcVecRowPitch[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowPitch, dstRowPitch, format.planePixelStrideBytes(0)};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowPitch, dstData->plane(0).buffer, dstData->plane(0).pitchBytes,
                               dstRowPitch, shape.y, cudaMemcpyDeviceToHost));

        // Generate gold result
        std::vector<float> kernel(kernelSize.w * kernelSize.h);

        int2  half{kernelSize.w / 2, kernelSize.h / 2};
        float sx  = 2.f * sigma.x * sigma.x;
        float sy  = 2.f * sigma.y * sigma.y;
        float s   = 2.f * sigma.x * sigma.y * M_PI;
        float sum = 0.f;
        for (int y = -half.y; y <= half.y; ++y)
        {
            for (int x = -half.x; x <= half.x; ++x)
            {
                float kv = std::exp(-((x * x) / sx + (y * y) / sy)) / s;

                kernel[(y + half.y) * kernelSize.w + (x + half.x)] = kv;

                sum += kv;
            }
        }
        for (int i = 0; i < kernelSize.w * kernelSize.h; ++i)
        {
            kernel[i] /= sum;
        }

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, format, kernel, kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}
