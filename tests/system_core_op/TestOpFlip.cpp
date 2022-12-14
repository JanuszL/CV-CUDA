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

#include <common/FlipUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
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
    {    123,     66,       2,    NVCV_IMAGE_FORMAT_U8,  1},
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
