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

#include <common/HashUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <list>
#include <random>

#include <nvcv/Fwd.hpp>

namespace nvcv = nv::cv;
namespace t    = ::testing;
namespace test = nv::cv::test;

class TensorTests
    : public t::TestWithParam<
          std::tuple<test::Param<"numImages", int>, test::Param<"width", int>, test::Param<"height", int>,
                     test::Param<"format", nvcv::ImageFormat>, test::Param<"shape", nvcv::TensorShape>,
                     test::Param<"dtype", nvcv::PixelType>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorTests,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::TensorShape, nvcv::PixelType>
    {
        {53, 32, 16, nvcv::FMT_RGBA8p, nvcv::TensorShape{{53, 4, 16, 32},nvcv::TensorLayout::NCHW} , nvcv::TYPE_U8},
        {14, 64, 18, nvcv::FMT_RGB8, nvcv::TensorShape{{14, 18, 64, 3},nvcv::TensorLayout::NHWC}, nvcv::TYPE_U8}
    }
);

// clang-format on

TEST_P(TensorTests, wip_create)
{
    const int               PARAM_NUM_IMAGES = std::get<0>(GetParam());
    const int               PARAM_WIDTH      = std::get<1>(GetParam());
    const int               PARAM_HEIGHT     = std::get<2>(GetParam());
    const nvcv::ImageFormat PARAM_FORMAT     = std::get<3>(GetParam());
    const nvcv::TensorShape GOLD_SHAPE       = std::get<4>(GetParam());
    const nvcv::PixelType   GOLD_DTYPE       = std::get<5>(GetParam());
    const int               GOLD_NDIM        = 4;

    nvcv::Tensor tensor(PARAM_NUM_IMAGES, {PARAM_WIDTH, PARAM_HEIGHT}, PARAM_FORMAT);

    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());
    EXPECT_EQ(GOLD_SHAPE, tensor.shape());
    EXPECT_EQ(GOLD_NDIM, tensor.ndim());
    EXPECT_EQ(GOLD_SHAPE.layout(), tensor.layout());
    ASSERT_NE(nullptr, tensor.handle());

    {
        const nvcv::ITensorData *data = tensor.exportData();
        ASSERT_NE(nullptr, data);

        ASSERT_EQ(tensor.dtype(), data->dtype());

        auto *devdata = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(data);
        ASSERT_NE(nullptr, devdata);

        EXPECT_EQ(GOLD_NDIM, devdata->ndim());
        ASSERT_EQ(GOLD_SHAPE, devdata->shape());
        ASSERT_EQ(GOLD_SHAPE.layout(), devdata->layout());
        ASSERT_EQ(GOLD_DTYPE, devdata->dtype());

        auto access = nvcv::TensorDataAccessPitchImagePlanar::Create(*devdata);
        ASSERT_TRUE(access);

        EXPECT_EQ(access->samplePitchBytes(), devdata->pitchBytes(0));
        EXPECT_EQ(access->planePitchBytes(), access->infoLayout().isChannelFirst() ? devdata->pitchBytes(1) : 0);
        EXPECT_EQ(access->numSamples(), devdata->shape(0));

        // Write data to each plane
        for (int i = 0; i < access->numSamples(); ++i)
        {
            void *sampleBuffer = access->sampleData(i);
            for (int p = 0; p < access->numPlanes(); ++p)
            {
                void *planeBuffer = access->planeData(p, sampleBuffer);

                ASSERT_EQ(cudaSuccess, cudaMemset2D(planeBuffer, access->rowPitchBytes(), i * 3 + p * 7,
                                                    access->numCols() * access->colPitchBytes(), access->numRows()))
                    << "Image #" << i << ", plane #" << p;
            }
        }

        // Check if no overwrites
        for (int i = 0; i < access->numSamples(); ++i)
        {
            void *sampleBuffer = access->sampleData(i);
            for (int p = 1; p < access->numPlanes(); ++p)
            {
                void *planeBuffer = access->planeData(p, sampleBuffer);

                // enough for one plane
                std::vector<uint8_t> buf(access->numCols() * access->colPitchBytes() * access->numRows());

                ASSERT_EQ(cudaSuccess,
                          cudaMemcpy2D(&buf[0], access->numCols() * access->colPitchBytes(), planeBuffer,
                                       access->rowPitchBytes(), access->numCols() * access->colPitchBytes(),
                                       access->numRows(), cudaMemcpyDeviceToHost))
                    << "Image #" << i << ", plane #" << p;

                ASSERT_TRUE(
                    all_of(buf.begin(), buf.end(), [gold = (uint8_t)(i * 3 + p * 7)](uint8_t v) { return v == gold; }))
                    << "Image #" << i << ", plane #" << p;
            }
        }
    }
}

TEST(TensorWrapData, wip_create)
{
    nvcv::ImageFormat fmt
        = nvcv::ImageFormat(nvcv::ColorModel::RGB, nvcv::CSPEC_BT601_ER, nvcv::MemLayout::PL, nvcv::DataType::FLOAT,
                            nvcv::Swizzle::S_XY00, nvcv::Packing::X16, nvcv::Packing::X16);
    nvcv::PixelType GOLD_DTYPE = fmt.planePixelType(0);

    nvcv::Tensor::Requirements reqs = nvcv::Tensor::CalcRequirements(5, {173, 79}, fmt);

    nvcv::TensorDataPitchDevice::Buffer buf;
    std::copy(reqs.pitchBytes, reqs.pitchBytes + NVCV_TENSOR_MAX_NDIM, buf.pitchBytes);
    // dummy value, just to check if memory won't be accessed internally. If it does,
    // it'll segfault.
    buf.data = reinterpret_cast<void *>(678);

    nvcv::TensorDataPitchDevice tdata(nvcv::TensorShape{reqs.shape, reqs.ndim, reqs.layout},
                                      nvcv::PixelType{reqs.dtype}, buf);

    auto access = nvcv::TensorDataAccessPitchImagePlanar::Create(tdata);
    ASSERT_TRUE(access);

    EXPECT_EQ(nvcv::TensorLayout::NCHW, tdata.layout());
    EXPECT_EQ(5, access->numSamples());
    EXPECT_EQ(173, access->numCols());
    EXPECT_EQ(79, access->numRows());
    EXPECT_EQ(2, access->numChannels());

    EXPECT_EQ(5, tdata.shape()[0]);
    EXPECT_EQ(173, tdata.shape()[3]);
    EXPECT_EQ(79, tdata.shape()[2]);
    EXPECT_EQ(2, tdata.shape()[1]);
    EXPECT_EQ(reinterpret_cast<void *>(678), tdata.data());
    EXPECT_EQ(4, tdata.ndim());

    nvcv::TensorWrapData tensor{tdata};

    ASSERT_NE(nullptr, tensor.handle());

    EXPECT_EQ(tdata.shape(), tensor.shape());
    EXPECT_EQ(tdata.layout(), tensor.layout());
    EXPECT_EQ(tdata.ndim(), tensor.ndim());
    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());

    const nvcv::ITensorData *data = tensor.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(data);
    ASSERT_NE(nullptr, devdata);

    auto accessRef = nvcv::TensorDataAccessPitchImagePlanar::Create(*devdata);
    ASSERT_TRUE(access);

    EXPECT_EQ(tdata.dtype(), devdata->dtype());
    EXPECT_EQ(tdata.shape(), devdata->shape());
    EXPECT_EQ(tdata.ndim(), devdata->ndim());

    EXPECT_EQ(tdata.data(), devdata->data());

    auto *mem = reinterpret_cast<std::byte *>(tdata.data());

    EXPECT_LE(mem + access->samplePitchBytes() * 4, accessRef->sampleData(4));
    EXPECT_LE(mem + access->samplePitchBytes() * 3, accessRef->sampleData(3));

    EXPECT_LE(mem + access->samplePitchBytes() * 4, accessRef->sampleData(4, accessRef->planeData(0)));
    EXPECT_LE(mem + access->samplePitchBytes() * 4 + access->planePitchBytes() * 1,
              accessRef->sampleData(4, accessRef->planeData(1)));

    EXPECT_LE(mem + access->samplePitchBytes() * 3, accessRef->sampleData(3, accessRef->planeData(0)));
    EXPECT_LE(mem + access->samplePitchBytes() * 3 + access->planePitchBytes() * 1,
              accessRef->sampleData(3, accessRef->planeData(1)));
}

class TensorWrapImageTests
    : public t::TestWithParam<
          std::tuple<test::Param<"size", nvcv::Size2D>, test::Param<"format", nvcv::ImageFormat>,
                     test::Param<"gold_shape", nvcv::TensorShape>, test::Param<"dtype", nvcv::PixelType>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorWrapImageTests,
    test::ValueList<nvcv::Size2D, nvcv::ImageFormat, nvcv::TensorShape, nvcv::PixelType>
    {
        {{61,23}, nvcv::FMT_RGBA8p, nvcv::TensorShape{{1,4,23,61},nvcv::TensorLayout::NCHW}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGBA8, nvcv::TensorShape{{1,23,61,4},nvcv::TensorLayout::NHWC}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGB8, nvcv::TensorShape{{1,23,61,3},nvcv::TensorLayout::NHWC}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_RGB8p, nvcv::TensorShape{{1,3,23,61},nvcv::TensorLayout::NCHW}, nvcv::TYPE_U8},
        {{61,23}, nvcv::FMT_F32, nvcv::TensorShape{{1,1,23,61},nvcv::TensorLayout::NCHW}, nvcv::TYPE_F32},
        {{61,23}, nvcv::FMT_2F32, nvcv::TensorShape{{1,23,61,2},nvcv::TensorLayout::NHWC}, nvcv::TYPE_F32},
    }
);

// clang-format on

TEST_P(TensorWrapImageTests, wip_create)
{
    const nvcv::Size2D      PARAM_SIZE   = std::get<0>(GetParam());
    const nvcv::ImageFormat PARAM_FORMAT = std::get<1>(GetParam());
    const nvcv::TensorShape GOLD_SHAPE   = std::get<2>(GetParam());
    const nvcv::PixelType   GOLD_DTYPE   = std::get<3>(GetParam());

    nvcv::Image img(PARAM_SIZE, PARAM_FORMAT);

    nvcv::TensorWrapImage tensor(img);

    EXPECT_EQ(GOLD_SHAPE, tensor.shape());
    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());

    auto *imgData    = dynamic_cast<const nvcv::IImageDataPitchDevice *>(img.exportData());
    auto *tensorData = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(tensor.exportData());

    auto tensorAccess = nvcv::TensorDataAccessPitchImagePlanar::Create(*tensorData);
    EXPECT_TRUE(tensorAccess);

    EXPECT_EQ(imgData->plane(0).buffer, tensorData->data());

    for (int p = 0; p < imgData->numPlanes(); ++p)
    {
        EXPECT_EQ(imgData->plane(p).buffer, tensorAccess->planeData(p));
        EXPECT_EQ(imgData->plane(p).pitchBytes, tensorAccess->rowPitchBytes());
        EXPECT_EQ(img.format().planePixelStrideBytes(p), tensorAccess->colPitchBytes());
    }
}
