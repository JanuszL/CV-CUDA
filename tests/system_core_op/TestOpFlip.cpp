/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <common/FlipUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpFlip.hpp>

#include <random>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;
namespace cuda = nv::cv::cuda;

// clang-format off

NVCV_TEST_SUITE_P(OpFlip, test::ValueList<int, int, int, NVCVImageFormat, int>
{
    // width, height, batches,                  format, flipCode
    {    176,    113,       1,    NVCV_IMAGE_FORMAT_U8,  0},
    {    123,     66,       5,    NVCV_IMAGE_FORMAT_U8,  1},
    {    123,     33,       3,  NVCV_IMAGE_FORMAT_RGB8, -1},
    {     42,     53,       4, NVCV_IMAGE_FORMAT_RGBA8,  1},
    {     13,     42,       3,  NVCV_IMAGE_FORMAT_RGB8,  0},
    {     62,    111,       4, NVCV_IMAGE_FORMAT_RGBA8, -1}
});

// clang-format on

TEST_P(OpFlip, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int flipCode = GetParamValue<4>();

    int3 shape{width, height, batches};

    nvcv::Tensor inTensor(batches, {width, height}, format);
    nvcv::Tensor outTensor(batches, {width, height}, format);

    const auto *input  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(inTensor.exportData());
    const auto *output = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(outTensor.exportData());

    ASSERT_NE(input, nullptr);
    ASSERT_NE(output, nullptr);

    long3 inPitches{input->pitchBytes(0), input->pitchBytes(1), input->pitchBytes(2)};
    long3 outPitches{output->pitchBytes(0), output->pitchBytes(1), output->pitchBytes(2)};

    long inBufSize  = inPitches.x * input->shape(0);
    long outBufSize = outPitches.x * output->shape(0);

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });
    std::vector<uint8_t> goldVec(outBufSize);
    test::FlipCPU(goldVec, outPitches, inVec, inPitches, shape, format, flipCode);

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(input->data(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    nv::cvop::Flip flipOp;
    EXPECT_NO_THROW(flipOp(stream, inTensor, outTensor, flipCode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // copy output back to host
    std::vector<uint8_t> testVec(outBufSize);
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), output->data(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(testVec, goldVec);
}

TEST_P(OpFlip, varshape_correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int flipCode = GetParamValue<4>();

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

    // Create flip code tensor
    nv::cv::Tensor flip_code({{batches}, "N"}, nv::cv::TYPE_S32);
    {
        auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(flip_code.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int> vec(batches, flipCode);

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(dev->data(), vec.data(), vec.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    }

    // Run operator
    nv::cvop::Flip flipOp(batches);

    EXPECT_NO_THROW(flipOp(stream, batchSrc, batchDst, flip_code));

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
        std::vector<uint8_t> goldVec(shape.y * pitches.y);
        test::FlipCPU(goldVec, pitches, srcVec[i], pitches, shape, format, flipCode);

        EXPECT_EQ(testVec, goldVec);
    }
}
