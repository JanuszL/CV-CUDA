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
#include "nvcv/DataLayout.hpp"
#include "nvcv/ImageFormat.hpp"
#include "nvcv/PixelType.hpp"

#include <common/BorderUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpCopyMakeBorder.hpp>

#include <random>

//#define DEBUG_PRINT_IMAGE
//#define DEBUG_PRINT_DIFF

namespace nvcv = nv::cv;
namespace test = nv::cv::test;

template<typename T>
static void CopyMakeBorder(std::vector<T> &hDst, const std::vector<T> &hSrc,
                           const nvcv::TensorDataAccessPitchImagePlanar &dDstData, const int srcWidth,
                           const int srcHeight, const int srcRowPitch, const int srcPixPitch, const int srcImgPitch,
                           const int top, const int left, const NVCVBorderType borderType, const float4 borderValue)
{
    int dstPixPitch = dDstData.numChannels();
    int dstRowPitch = dDstData.rowPitchBytes() / sizeof(T);
    int dstImgPitch = dDstData.samplePitchBytes() / sizeof(T);

    int2 coords, size{srcWidth, srcHeight};
    for (int db = 0; db < dDstData.numSamples(); db++)
    {
        for (int di = 0; di < dDstData.numRows(); di++)
        {
            coords.y = di - top;

            for (int dj = 0; dj < dDstData.numCols(); dj++)
            {
                coords.x = dj - left;

                for (int dk = 0; dk < dDstData.numChannels(); dk++)
                {
                    T out = 0;

                    if (coords.x >= 0 && coords.x < srcWidth && coords.y >= 0 && coords.y < srcHeight)
                    {
                        out = hSrc[db * srcImgPitch + coords.y * srcRowPitch + coords.x * srcPixPitch + dk];
                    }
                    else
                    {
                        if (borderType == NVCV_BORDER_CONSTANT)
                        {
                            out = static_cast<T>(reinterpret_cast<const float *>(&borderValue)[dk]);
                        }
                        else
                        {
                            if (borderType == NVCV_BORDER_REPLICATE)
                                test::ReplicateBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_WRAP)
                                test::WrapBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT)
                                test::ReflectBorderIndex(coords, size);
                            else if (borderType == NVCV_BORDER_REFLECT101)
                                test::Reflect101BorderIndex(coords, size);

                            out = hSrc[db * srcImgPitch + coords.y * srcRowPitch + coords.x * srcPixPitch + dk];
                        }
                    }

                    hDst[db * dstImgPitch + di * dstRowPitch + dj * dstPixPitch + dk] = out;
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpCopyMakeBorder, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float, float, float, float, nvcv::ImageFormat>
{
    // srcWidth, srcHeight, numBatches, topPad, buttomPad, leftPad, rightPad,         NVCVBorderType,    bValue1, bValue2, bValue3, bValue4, ImageFormat
    {       212,       113,          1,        0,         0,      0,       0,   NVCV_BORDER_CONSTANT,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {        12,        13,          2,       12,        16,      0,       3,   NVCV_BORDER_CONSTANT,       12.f,   100.f,   245.f,     0.f, nvcv::FMT_RGB8},
    {       212,       113,          3,        0,       113,      5,       0,   NVCV_BORDER_CONSTANT,       13.f,     5.f,     4.f,     0.f, nvcv::FMT_RGB8},
    {       212,       613,          4,       19,        20,      7,       7,   NVCV_BORDER_CONSTANT,      255.f,   255.f,   255.f,     0.f, nvcv::FMT_RGB8},

    {       234,       131,          2,       44,        55,     33,      22,  NVCV_BORDER_REPLICATE,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGB8},
    {       234,       131,          2,       33,        20,     41,      42,    NVCV_BORDER_REFLECT,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBA8},
    {       234,       131,          2,      100,        85,     53,      62,       NVCV_BORDER_WRAP,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBf32},
    {       243,       123,          2,       56,       123,     77,      98, NVCV_BORDER_REFLECT101,        0.f,     0.f,     0.f,     0.f, nvcv::FMT_RGBAf32},

});

// clang-format on

template<typename T>
void StartTest(int srcWidth, int srcHeight, int numBatches, int topPad, int buttomPad, int leftPad, int rightPad,
               NVCVBorderType borderType, float4 borderValue, nvcv::ImageFormat format)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int dstWidth  = srcWidth + leftPad + rightPad;
    int dstHeight = srcHeight + topPad + leftPad;

    std::vector<T> srcVec;

    nvcv::Tensor imgSrc(numBatches, {srcWidth, srcHeight}, format);
    const auto  *srcData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgSrc.exportData());
    ASSERT_NE(nullptr, srcData);
    auto srcAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*srcData);
    ASSERT_TRUE(srcData);
    int srcBufSize = (srcAccess->samplePitchBytes() / sizeof(T)) * srcAccess->numSamples();
    srcVec.resize(srcBufSize);

    std::default_random_engine             randEng{0};
    std::uniform_int_distribution<uint8_t> srcRand{0u, 255u};
    if (std::is_same<T, float>::value)
        std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng) / 255.0f; });
    else
        std::generate(srcVec.begin(), srcVec.end(), [&]() { return srcRand(randEng); });

    // Copy each input image with random data to the GPU
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(srcData->data(), srcVec.data(), srcBufSize * sizeof(T), cudaMemcpyHostToDevice, stream));

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, format);
    const auto  *dstData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());
    ASSERT_NE(nullptr, dstData);
    auto dstAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*dstData);
    ASSERT_TRUE(dstData);
    int dstBufSize = (dstAccess->samplePitchBytes() / sizeof(T)) * dstAccess->numSamples();
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->data(), 0, dstBufSize * sizeof(T), stream));

    std::vector<T> testVec(dstBufSize);
    std::vector<T> goldVec(dstBufSize);

    int srcPixPitch = srcAccess->numChannels();
    int srcRowPitch = srcAccess->rowPitchBytes() / sizeof(T);
    int srcImgPitch = srcAccess->samplePitchBytes() / sizeof(T);

    // Generate gold result
    CopyMakeBorder(goldVec, srcVec, *dstAccess, srcWidth, srcHeight, srcRowPitch, srcPixPitch, srcImgPitch, topPad,
                   leftPad, borderType, borderValue);

    // Generate test result
    nv::cvop::CopyMakeBorder cpyMakeBorderOp;

    EXPECT_NO_THROW(cpyMakeBorderOp(stream, imgSrc, imgDst, topPad, leftPad, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), dstData->data(), dstBufSize * sizeof(T), cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcPitchBytes / sizeof(uint8_t));
    test::DebugPrintImage(testVec, dstData->rowPitchBytes() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstData->rowPitchBytes() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec);
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}

TEST_P(OpCopyMakeBorder, correct_output)
{
    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int topPad     = GetParamValue<3>();
    int buttomPad  = GetParamValue<4>();
    int leftPad    = GetParamValue<5>();
    int rightPad   = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();
    float4         borderValue;
    borderValue.x = GetParamValue<8>();
    borderValue.y = GetParamValue<9>();
    borderValue.z = GetParamValue<10>();
    borderValue.w = GetParamValue<11>();

    nvcv::ImageFormat format = GetParamValue<12>();

    if (nvcv::FMT_RGB8 == format || nvcv::FMT_RGBA8 == format)
        StartTest<uint8_t>(srcWidth, srcHeight, numBatches, topPad, buttomPad, leftPad, rightPad, borderType,
                           borderValue, format);
    else if (nvcv::FMT_RGBf32 == format || nvcv::FMT_RGBAf32 == format)
        StartTest<float>(srcWidth, srcHeight, numBatches, topPad, buttomPad, leftPad, rightPad, borderType, borderValue,
                         format);
}
