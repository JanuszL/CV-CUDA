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
#include <common/TensorDataUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <nvcv/cuda/TypeTraits.hpp>
#include <operators/OpMorphology.hpp>

#include <random>

namespace nvcv = nv::cv;
namespace test = nv::cv::test;
namespace cuda = nv::cv::cuda;

using uchar = unsigned char;

// checks pixels only in the logical image region.
template<class T>
static bool imageRegionValuesSame(test::TensorImageData &a, test::TensorImageData &b)
{
    int minWidth  = a.size().w > b.size().w ? b.size().w : a.size().w;
    int minHeight = a.size().h > b.size().h ? b.size().h : a.size().h;

    if (a.bytesPerC() != b.bytesPerC() || a.imageCHW() != b.imageCHW() || a.numC() != b.numC())
        return false;

    for (int x = 0; x < minWidth; ++x)
        for (int y = 0; y < minHeight; ++y)
            for (int c = 0; c < a.numC(); ++c)
                if (*a.item<T>(x, y, c) != *b.item<T>(x, y, c))
                    return false;

    return true;
}

template<class T, size_t rows, size_t cols>
void SetTensorToTestVector(const uchar inputVals[rows][cols], int width, int height, nvcv::Tensor &tensor, int sample)
{
    test::TensorImageData data(tensor.exportData(), sample);

    for (int x = 0; x < width; ++x)
        for (int y = 0; y < height; ++y)
            for (int c = 0; c < data.numC(); ++c) *data.item<T>(x, y, c) = (T)inputVals[y][x];

    EXPECT_NO_THROW(test::SetTensorFromVector<T>(tensor.exportData(), data.getVector(), sample));
}

template<class T, size_t rows, size_t cols>
bool MatchTensorToTestVector(const uchar checkVals[rows][cols], int width, int height, nvcv::Tensor &Tensor, int sample)
{
    test::TensorImageData data(Tensor.exportData(), sample);
    for (int x = 0; x < width; ++x)
        for (int y = 0; y < height; ++y)
            for (int c = 0; c < data.numC(); ++c)
                if (*data.item<T>(x, y, c) != (T)checkVals[y][x])
                {
                    return false;
                }

    return true;
}

template<class T, size_t rows, size_t cols>
void checkTestVectors(cudaStream_t &stream, nvcv::Tensor &inTensor, nvcv::Tensor &outTensor,
                      const uchar input[rows][cols], const uchar output[rows][cols], int width, int height,
                      const nvcv::Size2D &maskSize, const int2 &anchor, int iteration, NVCVMorphologyType type,
                      NVCVBorderType borderMode, int batches)
{
    for (int i = 0; i < batches; ++i)
    {
        SetTensorToTestVector<uchar, rows, cols>(input, width, height, inTensor, i);
    }

    nv::cvop::Morphology morphOp;
    morphOp(stream, inTensor, outTensor, type, maskSize, anchor, iteration, borderMode);

    if (cudaSuccess != cudaStreamSynchronize(stream))
        throw std::runtime_error("Cuda Sync failed");

    for (int i = 0; i < batches; ++i)
    {
        if (MatchTensorToTestVector<uchar, rows, cols>(output, width, height, outTensor, i) != true)
        {
            throw std::runtime_error("Op returned unexpected result");
        }
    }
}

TEST(OpMorphology, morph_check_dilate_kernel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 5;
    constexpr int height  = 5;
    int           batches = 3;

    nvcv::ImageFormat  format{NVCV_IMAGE_FORMAT_U8};
    nvcv::Tensor       inTensor(batches, {width, height}, format);
    nvcv::Tensor       outTensor(batches, {width, height}, format);
    int2               anchor(-1, -1);
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_DILATE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width] ={
                        {0,0,0,0,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,0,0,0,0}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
    }

    // iteration = 2
    {
        // clang-format off
        iteration = 2;
        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width] ={
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1},
                        {1,1,1,1,1}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
        iteration = 1;
    }

    {
        // overlap
        // clang-format off
        uchar inImg[height][width] ={
                        {1,0,0,0,2},
                        {0,0,0,0,0},
                        {0,0,5,0,0},
                        {0,0,0,0,0},
                        {4,0,0,0,3}
                    };

        uchar expImg[height][width] ={
                        {1,1,0,2,2},
                        {1,5,5,5,2},
                        {0,5,5,5,0},
                        {4,5,5,5,3},
                        {4,4,0,3,3}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
    }

    {
        // mask
        // clang-format off
        maskSize.w = 1;
        maskSize.h = 2;
        uchar inImg[height][width] ={
                        {1,0,0,0,2},
                        {0,0,0,0,0},
                        {0,0,5,0,0},
                        {0,0,0,0,0},
                        {4,0,0,0,3}
                    };

        uchar expImg[height][width] ={
                        {1,0,0,0,2},
                        {1,0,0,0,2},
                        {0,0,5,0,0},
                        {0,0,5,0,0},
                        {4,0,0,0,3}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
        maskSize.w = 3;
        maskSize.h = 3;
    }

    // anchor
    {
        // clang-format off
        anchor.x = 0;
        anchor.y = 0;

        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width]  ={
                        {1,1,1,0,0},
                        {1,1,1,0,0},
                        {1,1,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
        anchor.x = -1;
        anchor.y = -1;
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMorphology, morph_check_erode_kernel)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 5;
    constexpr int height  = 5;
    int           batches = 3;

    nvcv::ImageFormat  format{NVCV_IMAGE_FORMAT_U8};
    nvcv::Tensor       inTensor(batches, {width, height}, format);
    nvcv::Tensor       outTensor(batches, {width, height}, format);
    int2               anchor(-1, -1);
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_ERODE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
        uchar inImg[height][width] ={
                        {0,0,0,0,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,1,1,1,0},
                        {0,0,0,0,0}
                    };

        uchar expImg[height][width] ={
                        {0,0,0,0,0},
                        {0,0,0,0,0},
                        {0,0,1,0,0},
                        {0,0,0,0,0},
                        {0,0,0,0,0}
                    };
        // clang-format on
        EXPECT_NO_THROW(
            (checkTestVectors<uchar, width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,
                                                    anchor, iteration, type, borderMode, batches)));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(OpMorphology, morph_check_dilate_kernel_even)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    constexpr int width   = 6;
    constexpr int height  = 6;
    int           batches = 3;

    nvcv::ImageFormat  format{NVCV_IMAGE_FORMAT_U8};
    nvcv::Tensor       inTensor(batches, {width, height}, format);
    nvcv::Tensor       outTensor(batches, {width, height}, format);
    int2               anchor(-1, -1);
    nvcv::Size2D       maskSize(3, 3);
    int                iteration  = 1;
    NVCVMorphologyType type       = NVCVMorphologyType::NVCV_DILATE;
    NVCVBorderType     borderMode = NVCVBorderType::NVCV_BORDER_CONSTANT;

    {
        // clang-format off
         uchar inImg[height][width] ={
                        {1,0,0,0,0,2},
                        {0,0,0,0,0,0},
                        {0,0,5,0,0,0},
                        {0,0,0,0,0,0},
                        {0,0,0,0,0,0},
                        {4,0,0,0,0,3}
                    };

        uchar expImg[height][width] ={
                        {1,1,0,0,2,2},
                        {1,5,5,5,2,2},
                        {0,5,5,5,0,0},
                        {0,5,5,5,0,0},
                        {4,4,0,0,3,3},
                        {4,4,0,0,3,3}
                    };

        EXPECT_NO_THROW((checkTestVectors<uchar,width, height>(stream, inTensor, outTensor, inImg, expImg, width, height, maskSize,anchor,iteration, type, borderMode, batches)));
    }
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

// clang-format off
NVCV_TEST_SUITE_P(OpMorphology, test::ValueList<int, int, int, NVCVImageFormat, int, int, NVCVBorderType, NVCVMorphologyType>
{
    // width, height, batches,                    format,  maskWidth, maskHeight,            borderMode, morphType
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_U8,          2,         2,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {      5,      5,       1,      NVCV_IMAGE_FORMAT_RGBAf32,     3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {    125,     35,       1,      NVCV_IMAGE_FORMAT_RGBA8,       3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {     52,     45,       1,      NVCV_IMAGE_FORMAT_U16,         3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {    325,     45,       3,      NVCV_IMAGE_FORMAT_RGB8,        3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},
    {     25,     45,       1,      NVCV_IMAGE_FORMAT_U8,          3,         3,   NVCV_BORDER_CONSTANT, NVCV_ERODE},
    {     25,     45,       2,      NVCV_IMAGE_FORMAT_U8,          3,         3,   NVCV_BORDER_CONSTANT, NVCV_DILATE},

});

// clang-format on

TEST_P(OpMorphology, morph_noop)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int                width      = GetParamValue<0>();
    int                height     = GetParamValue<1>();
    int                batches    = GetParamValue<2>();
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    nvcv::ImageFormat format{NVCV_IMAGE_FORMAT_U8};

    nvcv::Tensor inTensor(batches, {width, height}, format);
    nvcv::Tensor outTensor(batches, {width, height}, format);

    EXPECT_NO_THROW(test::SetTensorToRandomValue<uint8_t>(inTensor.exportData(), 0, 0xFF));
    EXPECT_NO_THROW(test::SetTensorTo<uint8_t>(outTensor.exportData(), 0));

    nv::cvop::Morphology morphOp;
    int2                 anchor(0, 0);

    nvcv::Size2D maskSize(1, 1);
    int          iteration = 0;
    EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, morphType, maskSize, anchor, iteration, borderMode));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    for (int i = 0; i < batches; ++i)
    {
        test::TensorImageData cvTensorIn(inTensor.exportData());
        test::TensorImageData cvTensorOut(outTensor.exportData());
        EXPECT_TRUE(imageRegionValuesSame<uint8_t>(cvTensorIn, cvTensorOut));
    }

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST_P(OpMorphology, morph_random)
{
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    int width   = GetParamValue<0>();
    int height  = GetParamValue<1>();
    int batches = GetParamValue<2>();

    nvcv::ImageFormat format{GetParamValue<3>()};

    nvcv::Size2D maskSize;
    maskSize.w                    = GetParamValue<4>();
    maskSize.h                    = GetParamValue<5>();
    NVCVBorderType     borderMode = GetParamValue<6>();
    NVCVMorphologyType morphType  = GetParamValue<7>();

    int  iteration = 1;
    int3 shape{width, height, batches};

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
    nv::cvop::Morphology morphOp;
    int2                 anchor(-1, -1);

    EXPECT_NO_THROW(morphOp(stream, inTensor, outTensor, morphType, maskSize, anchor, iteration, borderMode));

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));

    std::vector<uint8_t> goldVec(outBufSize);
    std::vector<uint8_t> testVec(outBufSize);

    // copy output back to host
    ASSERT_EQ(cudaSuccess, cudaMemcpy(testVec.data(), outData->data(), outBufSize, cudaMemcpyDeviceToHost));

    // generate gold result
    int2 kernelAnchor{maskSize.w / 2, maskSize.h / 2};
    test::Morph(goldVec, outPitches, inVec, inPitches, shape, format, maskSize, kernelAnchor, borderMode, morphType);

    EXPECT_EQ(testVec, goldVec);
}
