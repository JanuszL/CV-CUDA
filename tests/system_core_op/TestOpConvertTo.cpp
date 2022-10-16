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
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <nvcv/cuda/SaturateCast.hpp>
#include <operators/OpConvertTo.hpp>

#include <iostream>
#include <random>

namespace nvcv = nv::cv;
namespace gt   = ::testing;
namespace test = nv::cv::test;

template<typename DT_DEST>
static void setGoldBuffer(std::vector<DT_DEST> &vect, DT_DEST val, int width, int height, int rowStride, int imgStride,
                          int numImages)
{
    for (int img = 0; img < numImages; img++)
    {
        int imgStart = imgStride * img;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                vect[imgStart + x] = val;
            }
            imgStart += rowStride;
        }
    }
}

template<typename DT_SOURCE, typename DT_DEST>
const void testConvertTo(nvcv::ImageFormat fmtIn, nvcv::ImageFormat fmtOut, int batch, int width, int height,
                         double alpha, double beta, DT_SOURCE setVal, DT_DEST expVal)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    nvcv::Tensor imgOut(batch, {width, height}, fmtOut);
    nvcv::Tensor imgIn(batch, {width, height}, fmtIn);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgIn.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgOut.exportData());

    EXPECT_NE(nullptr, inData);
    EXPECT_NE(nullptr, outData);

    int inBufSizeElements  = (inData->imgPitchBytes() / sizeof(DT_SOURCE)) * inData->numImages();
    int outBufSizeElements = (outData->imgPitchBytes() / sizeof(DT_DEST)) * outData->numImages();
    int inBufSizeBytes     = inData->imgPitchBytes() * inData->numImages();
    int outBufSizeBytes    = outData->imgPitchBytes() * outData->numImages();

    std::vector<DT_SOURCE> srcVec(inBufSizeElements, setVal);
    std::vector<DT_DEST>   goldVec(outBufSizeElements);
    std::vector<DT_DEST>   testVec(outBufSizeElements);

    setGoldBuffer<DT_DEST>(goldVec, expVal, width * outData->dims().c, height,
                           (outData->rowPitchBytes() / sizeof(DT_DEST)), (outData->imgPitchBytes() / sizeof(DT_DEST)),
                           batch);

    // Copy input data to the GPU
    EXPECT_EQ(cudaSuccess,
              cudaMemcpyAsync(inData->mem(), srcVec.data(), inBufSizeBytes, cudaMemcpyHostToDevice, stream));
    EXPECT_EQ(cudaSuccess, cudaMemsetAsync(outData->mem(), 0x0, outBufSizeBytes, stream));

    // run operator
    nv::cvop::ConvertTo convertToOp;

    EXPECT_NO_THROW(convertToOp(stream, imgIn, imgOut, alpha, beta));
    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    EXPECT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->mem(), outBufSizeBytes, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    //dbgImage(goldVec, inData->rowPitchBytes());
    //dbgImage(testVec, outData->rowPitchBytes());
    EXPECT_EQ(goldVec, testVec);
}

// clang-format off
NVCV_TEST_SUITE_P(OpConvertTo, test::ValueList<int, int, double, double, int>
{
         //   width,     height,       alpha,          beta,   batch
         {       5,        5,          1.0,           1.0,       1 },
         {       5,        5,          2.1,           2.0,       5 },
         {       1,        1,          2.1,           -1.0,      1 }
});

// clang-format on

TEST_P(OpConvertTo, OpConvertTo_RGBA8toRGBA8)
{
    using toType   = uint8_t;
    using fromType = uint8_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBA8, nvcv::FMT_RGBA8, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBA8toRGBAf32)
{
    using toType   = float;
    using fromType = uint8_t;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBA8, nvcv::FMT_RGBAf32, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBAf32toRGBA8)
{
    using toType   = uint8_t;
    using fromType = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBAf32, nvcv::FMT_RGBA8, batch, width, height, alpha, beta, val, valExp);
}

TEST_P(OpConvertTo, OpConvertTo_RGBAf32toRGBAf32t)
{
    using toType   = float;
    using fromType = float;

    int    width  = GetParamValue<0>();
    int    height = GetParamValue<1>();
    double alpha  = GetParamValue<2>();
    double beta   = GetParamValue<3>();
    int    batch  = GetParamValue<4>();

    fromType val    = 0x10;
    toType   valExp = nvcv::cuda::SaturateCast<toType>(alpha * val + beta);

    testConvertTo<fromType, toType>(nvcv::FMT_RGBAf32, nvcv::FMT_RGBAf32, batch, width, height, alpha, beta, val,
                                    valExp);
}
