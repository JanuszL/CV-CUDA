/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Definitions.hpp"

#include <common/BorderUtils.hpp>
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <cvcuda/OpAdaptiveThreshold.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/DropCast.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <nvcv/cuda/TypeTraits.hpp>

#include <random>

namespace test = nvcv::test;
namespace cuda = nvcv::cuda;

using uchar = unsigned char;

// clang-format off

NVCV_TEST_SUITE_P(OpAdaptiveThreshold, test::ValueList<int, int, int, double, NVCVAdaptiveThresholdType, NVCVThresholdType, int, double>
{
    // width, height, batch, maxValue,      NVCVAdaptiveThresholdType,      NVCVThresholdType, blockSize,    c
    {    640,    360,     1,    100.0,    NVCV_ADAPTIVE_THRESH_MEAN_C,     NVCV_THRESH_BINARY,         3,  2.5},
    {   1280,    720,     2,    127.0,    NVCV_ADAPTIVE_THRESH_MEAN_C, NVCV_THRESH_BINARY_INV,         5, -4.3},
    {    123,     33,     3,    100.0,NVCV_ADAPTIVE_THRESH_GAUSSIAN_C,     NVCV_THRESH_BINARY,         7,  9.2},
    {    361,    768,     4,    127.0,NVCV_ADAPTIVE_THRESH_GAUSSIAN_C, NVCV_THRESH_BINARY_INV,         3, -2.8}
});

// clang-format on

// #define DBG 1

static void printVecDiff(const std::vector<uint8_t> &testVec, const std::vector<uint8_t> &goldVec)
{
#ifdef DBG
    int vecSize = testVec.size();
    std::cout << "vector size: " << vecSize << ", diff: " << std::endl;
    for (int i = 0; i < vecSize; ++i)
    {
        if (testVec[i] != goldVec[i])
        {
            std::cout << "diff i: " << i << ", testVec[i]: " << static_cast<int>(testVec[i])
                      << ", goldVec[i]: " << static_cast<int>(goldVec[i]) << "," << std::endl;
        }
    }
    std::cout << std::endl;
#endif
}

static std::vector<float> computeMeanKernel(int blockSize)
{
    int                kernelSize = blockSize * blockSize;
    std::vector<float> kernel(kernelSize, 1.0 / kernelSize);
    return kernel;
}

static std::vector<float> computeGaussianKernel(int blockSize)
{
    int                kernelSize = blockSize * blockSize;
    std::vector<float> kernel(kernelSize);

    int2 half{blockSize / 2, blockSize / 2};

    double2 sigma;
    sigma.x = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8;
    sigma.y = sigma.x;

    float sx  = 2.f * sigma.x * sigma.x;
    float sy  = 2.f * sigma.y * sigma.y;
    float s   = 2.f * sigma.x * sigma.y * M_PI;
    float sum = 0.f;
    for (int y = -half.y; y <= half.y; ++y)
    {
        for (int x = -half.x; x <= half.x; ++x)
        {
            float kv = std::exp(-((x * x) / sx + (y * y) / sy)) / s;

            kernel[(y + half.y) * blockSize + (x + half.x)] = kv;

            sum += kv;
        }
    }
    for (int i = 0; i < kernelSize; ++i)
    {
        kernel[i] /= sum;
    }
    return kernel;
}

template<typename T>
inline const T &ValueAt(const std::vector<uint8_t> &vec, int2 size, int y, int x)
{
    return *reinterpret_cast<const T *>(&vec[y * size.x + x]);
}

template<typename T>
inline T &ValueAt(std::vector<uint8_t> &vec, int2 size, int y, int x)
{
    return *reinterpret_cast<T *>(&vec[y * size.x + x]);
}

static void cpuAdaptiveThreshold(std::vector<uint8_t> &hDst, const std::vector<uint8_t> &hSrc, int2 size,
                                 const std::vector<float> &kernel, int blockSize, double maxValue,
                                 NVCVThresholdType thresholdType, double c)
{
    uchar iMaxValue = cuda::SaturateCast<uchar>(maxValue);
    int   idelta    = thresholdType == NVCV_THRESH_BINARY ? (int)std::ceil(c) : (int)std::floor(c);

    int r = blockSize / 2;
    for (int y = 0; y < size.y; ++y)
    {
        for (int x = 0; x < size.x; ++x)
        {
            float res = 0.f;

            int2 coord;

            for (int ky = 0; ky < blockSize; ++ky)
            {
                coord.y = y + ky - r;

                for (int kx = 0; kx < blockSize; ++kx)
                {
                    coord.x = x + kx - r;

                    test::ReplicateBorderIndex(coord, size);

                    uchar srcValue = ValueAt<uchar>(hSrc, size, coord.y, coord.x);

                    res += srcValue * kernel[ky * blockSize + kx];
                }
            }

            uchar srcV = ValueAt<uchar>(hSrc, size, y, x);
            uchar t;
            if (thresholdType == NVCV_THRESH_BINARY)
            {
                t = srcV + idelta > cuda::SaturateCast<uchar>(res) ? iMaxValue : 0;
            }
            else
            {
                t = srcV + idelta > cuda::SaturateCast<uchar>(res) ? 0 : iMaxValue;
            }

            ValueAt<uchar>(hDst, size, y, x) = t;
        }
    }
}

TEST_P(OpAdaptiveThreshold, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                       width                 = GetParamValue<0>();
    int                       height                = GetParamValue<1>();
    int                       batch                 = GetParamValue<2>();
    double                    maxValue              = GetParamValue<3>();
    NVCVAdaptiveThresholdType adaptiveThresholdType = GetParamValue<4>();
    NVCVThresholdType         thresholdType         = GetParamValue<5>();
    int                       blockSize             = GetParamValue<6>();
    double                    c                     = GetParamValue<7>();

    nvcv::ImageFormat fmt    = nvcv::FMT_U8;
    nvcv::Tensor      imgIn  = nvcv::test::CreateTensor(batch, width, height, fmt);
    nvcv::Tensor      imgOut = nvcv::test::CreateTensor(batch, width, height, fmt);

    auto inData = imgIn.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, inData);
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*inData);
    ASSERT_TRUE(inAccess);
    ASSERT_EQ(batch, inAccess->numSamples());

    auto outData = imgOut.exportData<nvcv::TensorDataStridedCuda>();
    ASSERT_NE(nullptr, outData);
    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);
    ASSERT_EQ(batch, outAccess->numSamples());

    int64_t outSampleStride = outAccess->sampleStride();

    if (outData->rank() == 3)
    {
        outSampleStride = outAccess->numRows() * outAccess->rowStride();
    }

    int64_t outBufferSize = outSampleStride * outAccess->numSamples();

    // Set output buffer to dummy value
    EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));

    //Generate input
    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::default_random_engine        randEng;
    int                               rowStride = width * fmt.planePixelStrideBytes(0);

    for (int i = 0; i < batch; i++)
    {
        std::uniform_int_distribution<uint8_t> rand(0, 255);
        srcVec[i].resize(height * rowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return rand(randEng); });
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(inAccess->sampleData(i), inAccess->rowStride(), srcVec[i].data(), rowStride,
                                            rowStride, height, cudaMemcpyHostToDevice));
    }

    // Call operator
    cvcuda::AdaptiveThreshold adaptiveThresholdOp(blockSize, 1);

    EXPECT_NO_THROW(
        adaptiveThresholdOp(stream, imgIn, imgOut, maxValue, adaptiveThresholdType, thresholdType, blockSize, c));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // generate gold result
    std::vector<float> kernel;
    if (adaptiveThresholdType == NVCV_ADAPTIVE_THRESH_MEAN_C)
    {
        kernel = computeMeanKernel(blockSize);
    }
    else
    {
        kernel = computeGaussianKernel(blockSize);
    }
#ifdef DBG
    std::cout << "kernel, len: " << kernel.size() << std::endl;
    for (size_t i = 0; i < kernel.size(); ++i)
    {
        std::cout << kernel[i] << ",";
    }
    std::cout << std::endl;
#endif

    for (int i = 0; i < batch; i++)
    {
        SCOPED_TRACE(i);

        int                  vecSize = height * rowStride;
        std::vector<uint8_t> testVec(vecSize);
        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), rowStride, outAccess->sampleData(i), outAccess->rowStride(),
                                            rowStride, height, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(vecSize);
        int2                 size{width, height};
        cpuAdaptiveThreshold(goldVec, srcVec[i], size, kernel, blockSize, maxValue, thresholdType, c);

        printVecDiff(testVec, goldVec);
        EXPECT_EQ(testVec, goldVec);
    }
}

TEST_P(OpAdaptiveThreshold, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                       width                 = GetParamValue<0>();
    int                       height                = GetParamValue<1>();
    int                       batch                 = GetParamValue<2>();
    double                    maxValue              = GetParamValue<3>();
    NVCVAdaptiveThresholdType adaptiveThresholdType = GetParamValue<4>();
    NVCVThresholdType         thresholdType         = GetParamValue<5>();
    int                       blockSize             = GetParamValue<6>();
    double                    c                     = GetParamValue<7>();

    nvcv::ImageFormat fmt = nvcv::FMT_U8;

    // Create input varshape
    std::default_random_engine         rng;
    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<std::unique_ptr<nvcv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(batch);
    std::vector<int>                  srcVecRowStride(batch);

    for (int i = 0; i < batch; ++i)
    {
        imgSrc.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{udistWidth(rng), udistHeight(rng)}, fmt));

        int srcRowStride   = imgSrc[i]->size().w * fmt.planePixelStrideBytes(0);
        srcVecRowStride[i] = srcRowStride;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i]->size().h * srcRowStride);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto imgData = imgSrc[i]->exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_NE(imgData, nvcv::NullOpt);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).basePtr, imgData->plane(0).rowStride, srcVec[i].data(),
                                    srcRowStride, srcRowStride, imgSrc[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nvcv::ImageBatchVarShape batchSrc(batch);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape
    std::vector<std::unique_ptr<nvcv::Image>> imgDst;
    for (int i = 0; i < batch; ++i)
    {
        imgDst.emplace_back(std::make_unique<nvcv::Image>(imgSrc[i]->size(), imgSrc[i]->format()));
    }
    nvcv::ImageBatchVarShape batchDst(batch);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create maxValue tensor
    nvcv::Tensor maxValueTensor({{batch}, "N"}, nvcv::TYPE_F64);
    {
        auto dev = maxValueTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<double> vec(batch, maxValue);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create blockSize tensor
    nvcv::Tensor blockSizeTensor({{batch}, "N"}, nvcv::TYPE_S32);
    {
        auto dev = blockSizeTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<int> vec(batch, blockSize);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(int),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Create c tensor
    nvcv::Tensor cTensor({{batch}, "N"}, nvcv::TYPE_F64);
    {
        auto dev = cTensor.exportData<nvcv::TensorDataStridedCuda>();
        ASSERT_NE(dev, nullptr);

        std::vector<double> vec(batch, c);

        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(dev->basePtr(), vec.data(), vec.size() * sizeof(double),
                                               cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    cvcuda::AdaptiveThreshold adaptiveThresholdOp(blockSize, batch);

    EXPECT_NO_THROW(adaptiveThresholdOp(stream, batchSrc, batchDst, maxValueTensor, adaptiveThresholdType,
                                        thresholdType, blockSizeTensor, cTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    std::vector<float> kernel;
    if (adaptiveThresholdType == NVCV_ADAPTIVE_THRESH_MEAN_C)
    {
        kernel = computeMeanKernel(blockSize);
    }
    else
    {
        kernel = computeGaussianKernel(blockSize);
    }
#ifdef DBG
    std::cout << "kernel, len: " << kernel.size() << std::endl;
    for (size_t i = 0; i < kernel.size(); ++i)
    {
        std::cout << kernel[i] << ",";
    }
    std::cout << std::endl;
#endif
    for (int i = 0; i < batch; ++i)
    {
        SCOPED_TRACE(i);

        const auto srcData = imgSrc[i]->exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto dstData = imgDst[i]->exportData<nvcv::ImageDataStridedCuda>();
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowStride = srcVecRowStride[i];

        int2 size{srcData->plane(0).width, srcData->plane(0).height};

        std::vector<uint8_t> testVec(size.y * dstRowStride);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess, cudaMemcpy2D(testVec.data(), dstRowStride, dstData->plane(0).basePtr,
                                            dstData->plane(0).rowStride, dstRowStride, size.y, cudaMemcpyDeviceToHost));

        // Generate gold result

        std::vector<uint8_t> goldVec(size.y * dstRowStride);

        cpuAdaptiveThreshold(goldVec, srcVec[i], size, kernel, blockSize, maxValue, thresholdType, c);

        printVecDiff(testVec, goldVec);
        EXPECT_EQ(testVec, goldVec);
    }
}
