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
#include <operators/OpCustomCrop.hpp>

#include <iostream>
#include <random>

namespace nvcv = nv::cv;
namespace gt   = ::testing;
namespace test = nv::cv::test;

//#define DBG_CROP_RECT

#ifdef DBG_CROP_RECT
static void dbgImage(std::vector<uint8_t> &in, int rowPitch)
{
    std::cout << "\n IMG -- " << rowPitch << " in " << in.size() << "\n";
    for (size_t i = 0; i < in.size(); i++)
    {
        if (i % rowPitch == 0)
            std::cout << "\n";
        printf("%02x,", in[i]);
    }
}
#endif

// Width is in bytes or pixels..
static void WriteData(const nvcv::ITensorDataPitchDevice *data, uint8_t val, NVCVRectI region)
{
    const std::array<int32_t, 4> &bpchan = data->format().bitsPerChannel();
    // all channels must be same size
    EXPECT_EQ(NVCV_TENSOR_NHWC, data->layout());
    EXPECT_EQ(true, std::all_of(bpchan.cbegin(), bpchan.cend(), [bpchan](int x) { return x == bpchan[0]; }));
    EXPECT_LE(region.x + region.width, data->dims().w);
    EXPECT_LE(region.y + region.height, data->dims().h);

    int      bytesPerChan  = data->format().bitsPerChannel()[0] / 8;
    int      bytesPerPixel = data->dims().c * bytesPerChan;
    uint8_t *impPtrTop     = (uint8_t *)data->mem();
    uint8_t *impPtr        = nullptr;
    int      numImages     = data->numImages();
    int      rowPitchBytes = data->rowPitchBytes();

    EXPECT_NE(nullptr, impPtrTop);
    for (int img = 0; img < numImages; img++)
    {
        impPtr = impPtrTop + (data->imgPitchBytes() * img) + (region.x * bytesPerPixel) + (rowPitchBytes * region.y);
        EXPECT_EQ(cudaSuccess,
                  cudaMemset2D((void *)impPtr, rowPitchBytes, val, region.width * bytesPerPixel, region.height));
    }
}

static void setGoldBuffer(std::vector<uint8_t> &vect, const nvcv::ITensorDataPitchDevice *data, NVCVRectI region,
                          uint8_t val)
{
    int bytesPerChan  = data->format().bitsPerChannel()[0] / 8;
    int bytesPerPixel = data->dims().c * bytesPerChan;

    uint8_t *ptrTop = vect.data();
    for (int img = 0; img < data->numImages(); img++)
    {
        uint8_t *ptr = ptrTop + data->imgPitchBytes() * img;
        for (int i = 0; i < region.height; i++)
        {
            memset(ptr, val, region.width * bytesPerPixel);
            ptr += data->rowPitchBytes();
        }
    }
}

// clang-format off
NVCV_TEST_SUITE_P(OpCustomCrop, test::ValueList<int, int, int, int, int, int, int, int, int>
{
    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       2,        2,        2,        2,          1,          1,     0,     0,            1},
    {       2,        2,        2,        2,          1,          1,     0,     1,            1},
    {       2,        2,        2,        2,          1,          1,     1,     0,            1},
    {       2,        2,        2,        2,          1,          1,     1,     1,            1},

    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       5,        5,        2,        2,          2,          2,     0,     0,            1},
    {       5,        5,        2,        2,          2,          2,     0,     1,            1},
    {       5,        5,        2,        2,          2,          2,     1,     0,            1},
    {       5,        5,        2,        2,          2,          2,     1,     1,            1},

    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       5,        5,        2,        2,          2,          2,     0,     0,            5},
    {       5,        5,        2,        2,          2,          2,     0,     3,            5},
    {       5,        5,        2,        2,          2,          2,     3,     0,            5},
    {       5,        5,        2,        2,          2,          2,     3,     3,            5},

    //inWidth, inHeight, outWidth, outHeight, cropWidth, cropHeight, cropX, cropY, numberImages
    {       5,        5,        5,        5,          1,          2,     0,     0,            2},
    {       5,        5,        5,        5,          1,          2,     0,     3,            2},
    {       5,        5,        5,        5,          1,          2,     4,     0,            2},
    {       5,        5,        5,        5,          1,          2,     4,     3,            2},

});

// clang-format on
TEST_P(OpCustomCrop, CustomCrop_packed)
{
    cudaStream_t stream;
    EXPECT_EQ(cudaSuccess, cudaStreamCreate(&stream));
    int     inWidth        = GetParamValue<0>();
    int     inHeight       = GetParamValue<1>();
    int     outWidth       = GetParamValue<2>();
    int     outHeight      = GetParamValue<3>();
    int     cropWidth      = GetParamValue<4>();
    int     cropHeight     = GetParamValue<5>();
    int     cropX          = GetParamValue<6>();
    int     cropY          = GetParamValue<7>();
    int     numberOfImages = GetParamValue<8>();
    uint8_t cropVal        = 0x56;

    nvcv::Tensor imgOut(numberOfImages, {outWidth, outHeight}, nvcv::FMT_RGBA8);
    nvcv::Tensor imgIn(numberOfImages, {inWidth, inHeight}, nvcv::FMT_RGBA8);

    const auto *inData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgIn.exportData());
    const auto *outData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgOut.exportData());

    EXPECT_NE(nullptr, inData);
    EXPECT_NE(nullptr, outData);

    int inBufSize
        = inData->imgPitchBytes() * inData->numImages(); //img pitch bytes can be more than the image 64, 128, etc
    int outBufSize = outData->imgPitchBytes() * outData->numImages();

    NVCVRectI crpRect = {cropX, cropY, cropWidth, cropHeight};

    EXPECT_EQ(cudaSuccess, cudaMemset(inData->mem(), 0x00, inData->imgPitchBytes() * inData->numImages()));
    EXPECT_EQ(cudaSuccess, cudaMemset(outData->mem(), 0x00, outData->imgPitchBytes() * outData->numImages()));
    WriteData(inData, cropVal, crpRect); // write data to be cropped

    std::vector<uint8_t> gold(outBufSize);
    setGoldBuffer(gold, outData, crpRect, cropVal);

    // run operator
    nv::cvop::CustomCrop cropOp;

    EXPECT_NO_THROW(cropOp(stream, imgIn, imgOut, crpRect));

    // check cdata
    std::vector<uint8_t> test(outBufSize);
    std::vector<uint8_t> testIn(inBufSize);

    EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(testIn.data(), inData->mem(), inBufSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(cudaSuccess, cudaMemcpy(test.data(), outData->mem(), outBufSize, cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));

#ifdef DBG_CROP_RECT
    dbgImage(testIn, inData->rowPitchBytes());
    dbgImage(test, outData->rowPitchBytes());
    dbgImage(gold, outData->rowPitchBytes());
#endif
    EXPECT_EQ(gold, test);
}