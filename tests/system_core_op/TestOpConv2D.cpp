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
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpConv2D.hpp>

#include <random>

namespace cuda = nv::cv::cuda;
namespace test = nv::cv::test;

// clang-format off

NVCV_TEST_SUITE_P(OpConv2D, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType>
{
    // width, height, numImages, kernelWidth, kernelHeight, kernelAnchorX, kernelAnchorY,           borderMode
    {     32,     33,         1,           3,            3,            -1,            -1, NVCV_BORDER_CONSTANT},
    {    123,    144,         2,           5,            5,            -1,            -1, NVCV_BORDER_CONSTANT},
    {     66,     99,         3,           7,            7,             5,             5, NVCV_BORDER_CONSTANT},
    {     13,     12,        13,           5,            5,             4,             4, NVCV_BORDER_WRAP},
    {      4,      3,         4,           3,            3,             1,             1, NVCV_BORDER_REPLICATE},
    {     44,     55,         5,           3,            3,            -1,            -1, NVCV_BORDER_REFLECT},
    {    244,    155,         6,           5,            5,            -1,            -1, NVCV_BORDER_REFLECT101}
});

// clang-format on

TEST_P(OpConv2D, varshape_correct_output)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width         = GetParamValue<0>();
    int height        = GetParamValue<1>();
    int numImages     = GetParamValue<2>();
    int kernelWidth   = GetParamValue<3>();
    int kernelHeight  = GetParamValue<4>();
    int kernelAnchorX = GetParamValue<5>();
    int kernelAnchorY = GetParamValue<6>();

    NVCVBorderType borderMode = GetParamValue<7>();

    nv::cv::ImageFormat imageFormat  = nv::cv::FMT_RGBA8;
    nv::cv::ImageFormat kernelFormat = nv::cv::FMT_F32;

    nv::cv::Size2D kernelSize{kernelWidth, kernelHeight};
    int2           kernelAnchor{kernelAnchorX, kernelAnchorY};

    float4 borderValue = cuda::SetAll<float4>(0);

    // Create input varshape

    std::default_random_engine rng;

    std::uniform_int_distribution<int> udistWidth(width * 0.8, width * 1.1);
    std::uniform_int_distribution<int> udistHeight(height * 0.8, height * 1.1);

    std::vector<std::unique_ptr<nv::cv::Image>> imgSrc;

    std::vector<std::vector<uint8_t>> srcVec(numImages);
    std::vector<int>                  srcVecRowPitch(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        imgSrc.emplace_back(
            std::make_unique<nv::cv::Image>(nv::cv::Size2D{udistWidth(rng), udistHeight(rng)}, imageFormat));

        int srcRowPitch   = imgSrc[i]->size().w * imageFormat.numChannels();
        srcVecRowPitch[i] = srcRowPitch;

        std::uniform_int_distribution<uint8_t> udist(0, 255);

        srcVec[i].resize(imgSrc[i]->size().h * srcRowPitch);
        std::generate(srcVec[i].begin(), srcVec[i].end(), [&]() { return udist(rng); });

        auto *imgData = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(imgSrc[i]->exportData());
        assert(imgData != nullptr);

        // Copy input data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(imgData->plane(0).buffer, imgData->plane(0).pitchBytes, srcVec[i].data(),
                                    srcRowPitch, srcRowPitch, imgSrc[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nv::cv::ImageBatchVarShape batchSrc(numImages);
    batchSrc.pushBack(imgSrc.begin(), imgSrc.end());

    // Create output varshape

    std::vector<std::unique_ptr<nv::cv::Image>> imgDst;
    for (int i = 0; i < numImages; ++i)
    {
        imgDst.emplace_back(std::make_unique<nv::cv::Image>(imgSrc[i]->size(), imgSrc[i]->format()));
    }
    nv::cv::ImageBatchVarShape batchDst(numImages);
    batchDst.pushBack(imgDst.begin(), imgDst.end());

    // Create kernel varshape

    std::vector<std::unique_ptr<nv::cv::Image>> kernel;
    std::vector<std::vector<float>>             kernelVec(numImages);

    for (int i = 0; i < numImages; ++i)
    {
        kernel.emplace_back(std::make_unique<nv::cv::Image>(kernelSize, kernelFormat));

        int pitchBytes = kernel[i]->size().w * sizeof(float);

        std::uniform_real_distribution<float> udist(0.f, 1.f);

        kernelVec[i].resize(kernel[i]->size().h * kernel[i]->size().w);

        std::generate(kernelVec[i].begin(), kernelVec[i].end(), [&]() { return udist(rng); });

        auto *data = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(kernel[i]->exportData());
        assert(data != nullptr);

        // Copy kernel data to the GPU
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2DAsync(data->plane(0).buffer, data->plane(0).pitchBytes, kernelVec[i].data(), pitchBytes,
                                    pitchBytes, kernel[i]->size().h, cudaMemcpyHostToDevice, stream));
    }

    nv::cv::ImageBatchVarShape batchKernel(numImages);
    batchKernel.pushBack(kernel.begin(), kernel.end());

    // Create kernel anchor tensor

    nv::cv::Tensor kernelAnchorTensor({{numImages}, "N"}, nv::cv::TYPE_2S32);

    {
        auto *dev = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(kernelAnchorTensor.exportData());
        ASSERT_NE(dev, nullptr);

        std::vector<int2> vec(numImages, kernelAnchor);

        ASSERT_EQ(cudaSuccess,
                  cudaMemcpyAsync(dev->data(), vec.data(), vec.size() * sizeof(int2), cudaMemcpyHostToDevice, stream));
    }

    // Generate test result

    nv::cvop::Conv2D conv2dOp;
    EXPECT_NO_THROW(conv2dOp(stream, batchSrc, batchDst, batchKernel, kernelAnchorTensor, borderMode));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    // Check test data against gold
    for (int i = 0; i < numImages; ++i)
    {
        SCOPED_TRACE(i);

        const auto *srcData = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(imgSrc[i]->exportData());
        ASSERT_EQ(srcData->numPlanes(), 1);

        const auto *dstData = dynamic_cast<const nv::cv::IImageDataPitchDevice *>(imgDst[i]->exportData());
        ASSERT_EQ(dstData->numPlanes(), 1);

        int dstRowPitch = srcVecRowPitch[i];

        int3  shape{srcData->plane(0).width, srcData->plane(0).height, 1};
        long3 pitches{shape.y * dstRowPitch, dstRowPitch, 4};

        std::vector<uint8_t> testVec(shape.y * pitches.y);

        // Copy output data to Host
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy2D(testVec.data(), dstRowPitch, dstData->plane(0).buffer, dstData->plane(0).pitchBytes,
                               dstRowPitch, shape.y, cudaMemcpyDeviceToHost));

        std::vector<uint8_t> goldVec(shape.y * pitches.y);

        // Generate gold result
        test::Convolve(goldVec, pitches, srcVec[i], pitches, shape, imageFormat, kernelVec[i], kernelSize, kernelAnchor,
                       borderMode, borderValue);

        EXPECT_EQ(testVec, goldVec);
    }
}
