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
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpReformat.hpp>

#include <iostream>
#include <random>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;
namespace cuda = nv::cv::cuda;

static void ReformatNHWC(std::vector<uint8_t> &hDst, long4 dstPitches, long4 dstShape,
                         const nvcv::TensorLayout &dstLayout, const std::vector<uint8_t> &hSrc, long4 srcPitches,
                         long4 srcShape, const nvcv::TensorLayout &srcLayout)
{
    long srcN = cuda::GetElement(srcShape, srcLayout.find('N'));
    long srcH = cuda::GetElement(srcShape, srcLayout.find('H'));
    long srcW = cuda::GetElement(srcShape, srcLayout.find('W'));
    long srcC = cuda::GetElement(srcShape, srcLayout.find('C'));
    long dstN = cuda::GetElement(dstShape, dstLayout.find('N'));
    long dstH = cuda::GetElement(dstShape, dstLayout.find('H'));
    long dstW = cuda::GetElement(dstShape, dstLayout.find('W'));
    long dstC = cuda::GetElement(dstShape, dstLayout.find('C'));

    EXPECT_EQ(srcN, dstN);
    EXPECT_EQ(srcH, dstH);
    EXPECT_EQ(srcW, dstW);
    EXPECT_EQ(srcC, dstC);

    long srcPbN = cuda::GetElement(srcPitches, srcLayout.find('N'));
    long srcPbH = cuda::GetElement(srcPitches, srcLayout.find('H'));
    long srcPbW = cuda::GetElement(srcPitches, srcLayout.find('W'));
    long srcPbC = cuda::GetElement(srcPitches, srcLayout.find('C'));
    long dstPbN = cuda::GetElement(dstPitches, dstLayout.find('N'));
    long dstPbH = cuda::GetElement(dstPitches, dstLayout.find('H'));
    long dstPbW = cuda::GetElement(dstPitches, dstLayout.find('W'));
    long dstPbC = cuda::GetElement(dstPitches, dstLayout.find('C'));

    for (long b = 0; b < dstN; ++b)
    {
        for (long y = 0; y < dstH; ++y)
        {
            for (long x = 0; x < dstW; ++x)
            {
                for (long c = 0; c < dstC; ++c)
                {
                    hDst[b * dstPbN + y * dstPbH + x * dstPbW + c * dstPbC]
                        = hSrc[b * srcPbN + y * srcPbH + x * srcPbW + c * srcPbC];
                }
            }
        }
    }
}

// Parameters are: width, height, batches, inFormat, outFormat

NVCV_TEST_SUITE_P(OpReformat, test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::ImageFormat>{
                                  {176, 113, 1,  nvcv::FMT_RGBA8,  nvcv::FMT_RGBA8},
                                  {156, 149, 1,   nvcv::FMT_RGB8,  nvcv::FMT_RGB8p},
                                  {222, 133, 1,  nvcv::FMT_RGB8p,   nvcv::FMT_RGB8},
                                  { 76,  13, 2,   nvcv::FMT_RGB8,   nvcv::FMT_RGB8},
                                  { 56,  49, 3,  nvcv::FMT_RGBA8, nvcv::FMT_RGBA8p},
                                  { 22,  33, 4, nvcv::FMT_RGBA8p,  nvcv::FMT_RGBA8}
});

TEST_P(OpReformat, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat inFormat  = GetParamValue<3>();
    nvcv::ImageFormat outFormat = GetParamValue<4>();

    nvcv::Tensor inTensor(batches, {width, height}, inFormat);
    nvcv::Tensor outTensor(batches, {width, height}, outFormat);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(inTensor.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(outTensor.exportData());

    ASSERT_NE(inData, nullptr);
    ASSERT_NE(outData, nullptr);

    long4 inPitches{inData->pitchBytes(0), inData->pitchBytes(1), inData->pitchBytes(2), inData->pitchBytes(3)};
    long4 outPitches{outData->pitchBytes(0), outData->pitchBytes(1), outData->pitchBytes(2), outData->pitchBytes(3)};

    long4 inShape{inData->shape(0), inData->shape(1), inData->shape(2), inData->shape(3)};
    long4 outShape{outData->shape(0), outData->shape(1), outData->shape(2), outData->shape(3)};

    long inBufSize  = inPitches.x * inShape.x;
    long outBufSize = outPitches.x * outShape.x;

    std::vector<uint8_t> inVec(inBufSize);

    std::default_random_engine    randEng(0);
    std::uniform_int_distribution rand(0u, 255u);

    std::generate(inVec.begin(), inVec.end(), [&]() { return rand(randEng); });

    // copy random input to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(inData->data(), inVec.data(), inBufSize, cudaMemcpyHostToDevice));

    // run operator
    nv::cvop::Reformat reformatOp;

    EXPECT_NO_THROW(reformatOp(stream, inTensor, outTensor));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->data(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    ReformatNHWC(goldVec, outPitches, outShape, outData->layout(), inVec, inPitches, inShape, inData->layout());

    EXPECT_EQ(testVec, goldVec);
}
