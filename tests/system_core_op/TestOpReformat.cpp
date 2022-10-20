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

#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpReformat.hpp>

#include <iostream>
#include <random>

namespace nvcv = nv::cv;

TEST(OpReformat, OpReformat_to_hwc)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut(1, {640, 480}, nvcv::FMT_RGB8);

    nvcv::Tensor::Requirements reqsPlanar = nvcv::Tensor::CalcRequirements(1, {640, 480}, nvcv::FMT_RGB8p);

    int64_t inBufferSize = CalcTotalSizeBytes(nvcv::Requirements{reqsPlanar.mem}.deviceMem());
    ASSERT_LT(0, inBufferSize);

    nvcv::TensorDataPitchDevice::Buffer bufPlanar;
    bufPlanar.ndim   = reqsPlanar.ndim;
    bufPlanar.dtype  = reqsPlanar.dtype;
    bufPlanar.layout = reqsPlanar.layout;
    std::copy(reqsPlanar.shape, reqsPlanar.shape + NVCV_TENSOR_MAX_NDIM, bufPlanar.shape);
    std::copy(reqsPlanar.pitchBytes, reqsPlanar.pitchBytes + NVCV_TENSOR_MAX_NDIM, bufPlanar.pitchBytes);
    EXPECT_EQ(cudaSuccess, cudaMalloc(&bufPlanar.mem, inBufferSize));

    nvcv::TensorDataPitchDevice bufIn(bufPlanar);

    ASSERT_EQ(1, bufIn.numImages());

    // setup the buffer
    for (int p = 0; p < bufIn.numPlanes(); p++)
    {
        //Set plane 0 to 0, 1 to 1's, 2, 2's etc.
        EXPECT_EQ(cudaSuccess, cudaMemset2D(bufIn.imgPlaneBuffer(0, p), bufIn.rowPitchBytes(), p,
                                            bufIn.dims().w * bufIn.colPitchBytes(), bufIn.dims().h));
    }

    // wrap the buffer
    nvcv::TensorWrapData imgIn{bufIn};

    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgOut.exportData());
    ASSERT_NE(nullptr, outData);

    // Set output buffer to dummy value
    EXPECT_EQ(cudaSuccess, cudaMemset(outData->mem(), 0xFA, outData->imgPitchBytes() * outData->numImages()));

    // Call operator
    nv::cvop::Reformat reformatOp;
    EXPECT_NO_THROW(reformatOp(stream, imgIn, imgOut));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    int64_t outBufferSize = outData->imgPitchBytes() * outData->numImages();

    std::vector<uint8_t> test(outBufferSize, 0xA);

    //Check data
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->mem(), outBufferSize, cudaMemcpyDeviceToHost));

    // plane data should be reordered in the buffer 0,1,2,3,0,1,2,3 ...
    EXPECT_EQ(test[0], 0);
    EXPECT_EQ(test[1], 1);
    EXPECT_EQ(test[2], 2);
    EXPECT_EQ(test[3], 0);
    EXPECT_EQ(test[4], 1);

    EXPECT_EQ(cudaSuccess, cudaFree(bufPlanar.mem));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpReformat, wip_OpReformat_same)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut(1, {640, 480}, nvcv::FMT_RGBA8);
    nvcv::Tensor imgIn(1, {640, 480}, nvcv::FMT_RGBA8);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgIn.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgOut.exportData());

    EXPECT_NE(nullptr, inData);
    EXPECT_NE(nullptr, outData);

    int inBufSize  = inData->imgPitchBytes() * inData->numImages();
    int outBufSize = outData->imgPitchBytes() * outData->numImages();

    EXPECT_EQ(inBufSize, outBufSize);

    std::vector<uint8_t>          gold(outBufSize);
    std::default_random_engine    rng;
    std::uniform_int_distribution rand;

    generate(gold.begin(), gold.end(), [&rng, &rand] { return rand(rng); });

    EXPECT_EQ(cudaSuccess, cudaMemcpy(inData->mem(), gold.data(), gold.size(), cudaMemcpyHostToDevice));

    // run operator
    nv::cvop::Reformat reformatOp;

    EXPECT_NO_THROW(reformatOp(stream, imgIn, imgOut));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // check cdata
    std::vector<uint8_t> test(outBufSize);

    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->mem(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    EXPECT_EQ(gold, test);
}
