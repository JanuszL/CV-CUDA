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
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/alloc/CustomAllocator.hpp>
#include <nvcv/alloc/CustomResourceAllocator.hpp>
#include <operators/OpPadAndStack.hpp>

//#define DEBUG_PRINT_IMAGE
//#define DEBUG_PRINT_DIFF

#include <common/Utils.hpp>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;

static int ReflectBorderIndex(int x, int size, const NVCVBorderType borderType)
{
    int delta = borderType == NVCV_BORDER_REFLECT101 ? 1 : 0;
    do
    {
        if (x < 0)
        {
            x = -x - 1 + delta;
        }
        else
        {
            x = size - 1 - (x - size) - delta;
        }
    }
    while (x < 0 || x >= size);
    return x;
}

static void PadAndStack(std::vector<uint8_t> &hDst, const std::vector<std::vector<uint8_t>> &hBatchSrc,
                        const nvcv::ITensorDataPitchDevice *dDstData, const int srcWidth, const int srcHeight,
                        const int srcRowPitch, const int srcPixPitch, const std::vector<int> &topVec,
                        const std::vector<int> &leftVec, const NVCVBorderType borderType, const float borderValue)
{
    int dstPixPitch = dDstData->dims().c;
    int dstRowPitch = dDstData->rowPitchBytes() / sizeof(uint8_t);
    int dstImgPitch = dDstData->imgPitchBytes() / sizeof(uint8_t);

    for (int db = 0; db < dDstData->dims().n; db++)
    {
        for (int di = 0; di < dDstData->dims().h; di++)
        {
            int si = di - topVec[db];

            for (int dj = 0; dj < dDstData->dims().w; dj++)
            {
                int sj = dj - leftVec[db];

                for (int dk = 0; dk < dDstData->dims().c; dk++)
                {
                    uint8_t out = 0;

                    if (si >= 0 && si < srcHeight && sj >= 0 && sj < srcWidth)
                    {
                        out = hBatchSrc[db][si * srcRowPitch + sj * srcPixPitch + dk];
                    }
                    else
                    {
                        if (borderType == NVCV_BORDER_CONSTANT)
                        {
                            out = static_cast<uint8_t>(borderValue);
                        }
                        else
                        {
                            if (borderType == NVCV_BORDER_REPLICATE)
                            {
                                si = std::max(0, std::min(srcHeight, si));
                                sj = std::max(0, std::min(srcWidth, sj));
                            }
                            else if (borderType == NVCV_BORDER_WRAP)
                            {
                                si = (si >= 0) ? si : (si - ((si - srcHeight + 1) / srcHeight) * srcHeight);
                                si = (si < srcHeight) ? si : (si % srcHeight);

                                sj = (sj >= 0) ? sj : (sj - ((sj - srcWidth + 1) / srcWidth) * srcWidth);
                                sj = (sj < srcWidth) ? sj : (sj % srcWidth);
                            }
                            else if (borderType == NVCV_BORDER_REFLECT || borderType == NVCV_BORDER_REFLECT101)
                            {
                                si = ReflectBorderIndex(si, srcHeight, borderType);
                                sj = ReflectBorderIndex(sj, srcWidth, borderType);
                            }

                            out = hBatchSrc[db][si * srcRowPitch + sj * srcPixPitch + dk];
                        }
                    }

                    hDst[db * dstImgPitch + di * dstRowPitch + dj * dstPixPitch + dk] = out;
                }
            }
        }
    }
}

// clang-format off

NVCV_TEST_SUITE_P(OpPadAndStack, test::ValueList<int, int, int, int, int, int, int, NVCVBorderType, float>
{
    // srcWidth, srcHeight, numBatches, dstWidth, dstHeight, topPad, leftPad,         NVCVBorderType, borderValue
    {       212,       113,          1,      111,       132,      0,       0,   NVCV_BORDER_CONSTANT,         0.f},
    {        12,        13,          2,      211,       232,      0,       3,   NVCV_BORDER_CONSTANT,        12.f},
    {       212,       113,          3,       11,       432,      5,       0,   NVCV_BORDER_CONSTANT,        13.f},
    {       212,       613,          4,      311,       532,      7,       7,   NVCV_BORDER_CONSTANT,       134.f},

    {       234,       131,          2,      131,       130,     33,      22,  NVCV_BORDER_REPLICATE,         0.f},
    {       234,       131,          2,      123,       132,     41,      42,    NVCV_BORDER_REFLECT,         0.f},
    {       234,       131,          2,      134,       131,     53,      62,       NVCV_BORDER_WRAP,         0.f},
    {       243,       123,          2,      132,       123,     77,      98, NVCV_BORDER_REFLECT101,         0.f},

});

// clang-format on

TEST_P(OpPadAndStack, correct_output)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int srcWidth   = GetParamValue<0>();
    int srcHeight  = GetParamValue<1>();
    int numBatches = GetParamValue<2>();
    int dstWidth   = GetParamValue<3>();
    int dstHeight  = GetParamValue<4>();
    int topPad     = GetParamValue<5>();
    int leftPad    = GetParamValue<6>();

    NVCVBorderType borderType = GetParamValue<7>();

    float borderValue = GetParamValue<8>();

    nvcv::Tensor inTop(1, {numBatches, 1}, nvcv::FMT_S32);
    nvcv::Tensor inLeft(1, {numBatches, 1}, nvcv::FMT_S32);

    const auto *inTopData  = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(inTop.exportData());
    const auto *inLeftData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(inLeft.exportData());

    ASSERT_NE(nullptr, inTopData);
    ASSERT_NE(nullptr, inLeftData);

    int inTopBufSize  = (inTopData->imgPitchBytes() / sizeof(int)) * inTopData->numImages();
    int inLeftBufSize = (inLeftData->imgPitchBytes() / sizeof(int)) * inLeftData->numImages();

    ASSERT_EQ(inTopBufSize, inLeftBufSize);

    std::vector<int> topVec(inTopBufSize);
    std::vector<int> leftVec(inLeftBufSize);

    for (int b = 0; b < numBatches; ++b)
    {
        topVec[b]  = topPad;
        leftVec[b] = leftPad;
    }

    // Copy vectors with top and left padding to the GPU
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inTopData->data(), topVec.data(), topVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(inLeftData->data(), leftVec.data(), leftVec.size() * sizeof(int),
                                           cudaMemcpyHostToDevice, stream));

    std::vector<std::unique_ptr<nvcv::IImage>> srcImgVec;

    std::vector<std::vector<uint8_t>> batchSrcVec;

    int srcPitchBytes = 0, srcRowPitch = 0, srcPixPitch = 0;

    for (int b = 0; b < numBatches; ++b)
    {
        srcImgVec.emplace_back(std::make_unique<nvcv::Image>(nvcv::Size2D{srcWidth, srcHeight}, nvcv::FMT_RGBA8));

        auto *imgSrcData = dynamic_cast<const nvcv::IImageDataDevicePitch *>(srcImgVec.back()->exportData());

        srcPitchBytes  = imgSrcData->plane(0).pitchBytes;
        srcRowPitch    = srcPitchBytes / sizeof(uint8_t);
        srcPixPitch    = 4;
        int srcBufSize = srcRowPitch * imgSrcData->plane(0).height;

        std::vector<uint8_t> srcVec(srcBufSize);

        test::FillRandomData(srcVec);

        // Copy each input image with random data to the GPU
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(imgSrcData->plane(0).buffer, srcVec.data(),
                                               srcVec.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

        batchSrcVec.push_back(srcVec);
    }

    nvcv::ImageBatchVarShape imgBatchSrc(numBatches, nvcv::FMT_RGBA8);

    imgBatchSrc.pushBack(srcImgVec.begin(), srcImgVec.end());

    nvcv::Tensor imgDst(numBatches, {dstWidth, dstHeight}, nvcv::FMT_RGBA8);

    const auto *dstData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(imgDst.exportData());

    ASSERT_NE(nullptr, dstData);

    int dstBufSize = (dstData->imgPitchBytes() / sizeof(uint8_t)) * dstData->numImages();

    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(dstData->data(), 0, dstBufSize * sizeof(uint8_t), stream));

    std::vector<uint8_t> testVec(dstBufSize);
    std::vector<uint8_t> goldVec(dstBufSize);

    // Generate gold result
    PadAndStack(goldVec, batchSrcVec, dstData, srcWidth, srcHeight, srcRowPitch, srcPixPitch, topVec, leftVec,
                borderType, borderValue);

    // Generate test result
    nv::cvop::PadAndStack padAndStackOp;

    EXPECT_NO_THROW(padAndStackOp(stream, imgBatchSrc, imgDst, inTop, inLeft, borderType, borderValue));

    // Get test data back
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), dstData->data(), dstBufSize, cudaMemcpyDeviceToHost));

#ifdef DEBUG_PRINT_IMAGE
    for (int b = 0; b < numBatches; ++b) test::DebugPrintImage(batchSrcVec[b], srcPitchBytes / sizeof(uint8_t));
    test::DebugPrintImage(testVec, dstData->rowPitchBytes() / sizeof(uint8_t));
    test::DebugPrintImage(goldVec, dstData->rowPitchBytes() / sizeof(uint8_t));
#endif
#ifdef DEBUG_PRINT_DIFF
    if (goldVec != testVec)
    {
        test::DebugPrintDiff(testVec, goldVec, dstData->rowPitchBytes() / sizeof(uint8_t),
                             dstData->rowPitchBytes() / sizeof(uint8_t));
    }
#endif

    EXPECT_EQ(goldVec, testVec);
}
