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
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpErase.hpp>

#include <iostream>

namespace nvcv = nv::cv;

TEST(OpErase, OpErase_Tensor)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut(1, {640, 480}, nvcv::FMT_U8);

    nvcv::Tensor::Requirements reqsPlanar = nvcv::Tensor::CalcRequirements(1, {640, 480}, nvcv::FMT_U8);

    int64_t inBufferSize = CalcTotalSizeBytes(nvcv::Requirements{reqsPlanar.mem}.deviceMem());
    ASSERT_LT(0, inBufferSize);

    nvcv::TensorDataPitchDevice::Buffer bufPlanar;
    std::copy(reqsPlanar.pitchBytes, reqsPlanar.pitchBytes + NVCV_TENSOR_MAX_NDIM, bufPlanar.pitchBytes);
    EXPECT_EQ(cudaSuccess, cudaMalloc(&bufPlanar.data, inBufferSize));

    nvcv::TensorDataPitchDevice bufIn(nvcv::TensorShape{reqsPlanar.shape, reqsPlanar.ndim, reqsPlanar.layout},
                                      nvcv::PixelType{reqsPlanar.dtype}, bufPlanar);

    auto inAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(bufIn);
    ASSERT_TRUE(inAccess);

    ASSERT_EQ(1, inAccess->numSamples());

    // setup the buffer
    EXPECT_EQ(cudaSuccess, cudaMemset2D(inAccess->planeData(0), inAccess->rowPitchBytes(), 0,
                                        inAccess->numCols() * inAccess->colPitchBytes(), inAccess->numRows()));

    // wrap the buffer
    nvcv::TensorWrapData imgIn{bufIn};

    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgOut.exportData());
    ASSERT_NE(nullptr, outData);

    auto outAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*outData);
    ASSERT_TRUE(outAccess);

    int64_t outBufferSize = outAccess->samplePitchBytes() * outAccess->numSamples();

    // Set output buffer to dummy value
    EXPECT_EQ(cudaSuccess, cudaMemset(outAccess->sampleData(0), 0xFA, outBufferSize));

    //parameters
    nvcv::Tensor anchor_x(1, {1, 1}, nvcv::FMT_S32);
    nvcv::Tensor anchor_y(1, {1, 1}, nvcv::FMT_S32);
    nvcv::Tensor erasing_w(1, {1, 1}, nvcv::FMT_S32);
    nvcv::Tensor erasing_h(1, {1, 1}, nvcv::FMT_S32);
    nvcv::Tensor erasing_c(1, {1, 1}, nvcv::FMT_S32);
    nvcv::Tensor values(1, {1, 1}, nvcv::FMT_F32);
    nvcv::Tensor imgIdx(1, {1, 1}, nvcv::FMT_S32);

    const auto *anchorxData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(anchor_x.exportData());
    const auto *anchoryData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(anchor_y.exportData());
    const auto *erasingwData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(erasing_w.exportData());
    const auto *erasinghData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(erasing_h.exportData());
    const auto *erasingcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(erasing_c.exportData());
    const auto *valuesData   = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(values.exportData());
    const auto *imgIdxData   = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgIdx.exportData());

    ASSERT_NE(nullptr, anchorxData);
    ASSERT_NE(nullptr, anchoryData);
    ASSERT_NE(nullptr, erasingwData);
    ASSERT_NE(nullptr, erasinghData);
    ASSERT_NE(nullptr, erasingcData);
    ASSERT_NE(nullptr, valuesData);
    ASSERT_NE(nullptr, imgIdxData);

    auto anchorxAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*anchorxData);
    ASSERT_TRUE(anchorxAccess);
    auto anchoryAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*anchoryData);
    ASSERT_TRUE(anchoryAccess);
    auto erasingwAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*erasingwData);
    ASSERT_TRUE(erasingwAccess);
    auto erasinghAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*erasinghData);
    ASSERT_TRUE(erasinghAccess);
    auto erasingcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*erasingcData);
    ASSERT_TRUE(erasingcAccess);
    auto valuesAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*valuesData);
    ASSERT_TRUE(valuesAccess);
    auto imgIdxAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*imgIdxData);
    ASSERT_TRUE(imgIdxAccess);

    int intBufSize   = (anchorxAccess->samplePitchBytes() / sizeof(int)) * anchorxAccess->numSamples();
    int floatBufSize = (valuesAccess->samplePitchBytes() / sizeof(float)) * valuesAccess->numSamples();

    ASSERT_EQ(intBufSize, floatBufSize);

    std::vector<int>   anchorxVec(intBufSize);
    std::vector<int>   anchoryVec(intBufSize);
    std::vector<int>   erasingwVec(intBufSize);
    std::vector<int>   erasinghVec(intBufSize);
    std::vector<int>   erasingcVec(intBufSize);
    std::vector<int>   imgIdxVec(intBufSize);
    std::vector<float> valuesVec(floatBufSize);

    anchorxVec[0]  = 0;
    anchoryVec[0]  = 0;
    erasingwVec[0] = 10;
    erasinghVec[0] = 10;
    erasingcVec[0] = 0x1;
    imgIdxVec[0]   = 0;
    valuesVec[0]   = 1.f;

    // Copy vectors to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchorxData->data(), anchorxVec.data(), anchorxVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(anchoryData->data(), anchoryVec.data(), anchoryVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingwData->data(), erasingwVec.data(), erasingwVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasinghData->data(), erasinghVec.data(), erasinghVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(erasingcData->data(), erasingcVec.data(), erasingcVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgIdxData->data(), imgIdxVec.data(), imgIdxVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(valuesData->data(), valuesVec.data(), valuesVec.size() * sizeof(float),
                                           cudaMemcpyHostToDevice, stream));

    // Call operator
    int             max_eh = 10, max_ew = 10;
    unsigned int    seed   = 0;
    bool            random = false, inplace = false;
    nv::cvop::Erase eraseOp;
    EXPECT_NO_THROW(eraseOp(stream, imgIn, imgOut, anchor_x, anchor_y, erasing_w, erasing_h, erasing_c, values, imgIdx,
                            max_eh, max_ew, random, seed, inplace));

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    std::vector<uint8_t> test(outBufferSize, 0xA);

    //Check data
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->data(), outBufferSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(test[0], 1);
    EXPECT_EQ(test[9], 1);
    EXPECT_EQ(test[10], 0);
    EXPECT_EQ(test[9 * 640], 1);
    EXPECT_EQ(test[9 * 640 + 9], 1);
    EXPECT_EQ(test[9 * 640 + 10], 0);
    EXPECT_EQ(test[10 * 640], 0);
    EXPECT_EQ(test[10 * 640 + 10], 0);

    EXPECT_EQ(cudaSuccess, cudaFree(bufPlanar.data));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
