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
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpAverageBlur.hpp>

#include <random>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;
namespace cuda = nv::cv::cuda;

// clang-format off

NVCV_TEST_SUITE_P(OpAverageBlur, test::ValueList<int, int, int, NVCVImageFormat, int, int, int, int, NVCVBorderType>
{
    // width, height, batches,                    format, ksizeX, ksizeY, kanchorX, kanchorY,           borderMode
    {    176,    113,       1,      NVCV_IMAGE_FORMAT_U8,      3,      3,       -1,       -1, NVCV_BORDER_CONSTANT},
    {    123,     66,       2,      NVCV_IMAGE_FORMAT_U8,      5,      5,        2,        2, NVCV_BORDER_CONSTANT},
    {    123,     33,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,        2,        2, NVCV_BORDER_WRAP},
    {     42,     53,       4,   NVCV_IMAGE_FORMAT_RGBA8,      7,      7,        5,        5, NVCV_BORDER_REPLICATE},
    {     13,     42,       3,    NVCV_IMAGE_FORMAT_RGB8,      3,      3,        1,        1, NVCV_BORDER_REFLECT},
    {     62,    111,       4,   NVCV_IMAGE_FORMAT_RGBA8,      9,      9,        8,        8, NVCV_BORDER_REFLECT101}
});

// clang-format on

TEST_P(OpAverageBlur, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    int ksizeX   = GetParamValue<4>();
    int ksizeY   = GetParamValue<5>();
    int kanchorX = GetParamValue<6>();
    int kanchorY = GetParamValue<7>();

    NVCVBorderType borderMode = GetParamValue<8>();

    float4 borderValue = cuda::SetAll<float4>(0);

    int3 shape{width, height, batches};

    nvcv::Size2D kernelSize(ksizeX, ksizeY);

    int2 kernelAnchor{kanchorX, kanchorY};

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
    nv::cvop::AverageBlur averageBlurOp(kernelSize);

    EXPECT_NO_THROW(averageBlurOp(stream, inTensor, outTensor, kernelSize, kernelAnchor, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->data(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    std::size_t ks = kernelSize.w * kernelSize.h;
    float       kv = 1.f / ks;

    std::vector<float> kernel(ks, kv);

    test::Convolve(goldVec, outPitches, inVec, inPitches, shape, format, kernel, kernelSize, kernelAnchor, borderMode,
                   borderValue);

    EXPECT_EQ(testVec, goldVec);
}
