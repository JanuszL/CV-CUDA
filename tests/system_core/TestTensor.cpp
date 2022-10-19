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
#include <nvcv/Tensor.hpp>

#include <list>
#include <random>

namespace nvcv = nv::cv;
namespace t    = ::testing;
namespace test = nv::cv::test;

class TensorTests
    : public t::TestWithParam<std::tuple<test::Param<"numImages", int>, test::Param<"width", int>,
                                         test::Param<"height", int>, test::Param<"format", nvcv::ImageFormat>,
                                         test::Param<"layout", nvcv::TensorLayout>, test::Param<"shape", nvcv::Shape>>>
{
};

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(_, TensorTests,
    test::ValueList<int, int, int, nvcv::ImageFormat, nvcv::TensorLayout, nvcv::Shape>
    {
        {53, 32, 16, nvcv::FMT_RGBA8p, nvcv::TensorLayout::NCHW, nvcv::Shape{53, 4, 16, 32}},
        {14, 64, 18, nvcv::FMT_RGB8, nvcv::TensorLayout::NHWC, nvcv::Shape{14, 18, 64, 3}}
    }
);

// clang-format on

TEST_P(TensorTests, wip_create)
{
    const int                PARAM_NUM_IMAGES = std::get<0>(GetParam());
    const int                PARAM_WIDTH      = std::get<1>(GetParam());
    const int                PARAM_HEIGHT     = std::get<2>(GetParam());
    const nvcv::ImageFormat  PARAM_FORMAT     = std::get<3>(GetParam());
    const nvcv::TensorLayout GOLD_LAYOUT      = std::get<4>(GetParam());
    const nvcv::Shape        GOLD_SHAPE       = std::get<5>(GetParam());
    const nvcv::DimsNCHW     GOLD_DIMS
        = nvcv::DimsNCHW{PARAM_NUM_IMAGES, PARAM_FORMAT.numChannels(), PARAM_HEIGHT, PARAM_WIDTH};
    const int GOLD_NDIM = 4;

    nvcv::Tensor tensor(PARAM_NUM_IMAGES, {PARAM_WIDTH, PARAM_HEIGHT}, PARAM_FORMAT);

    EXPECT_EQ(PARAM_FORMAT, tensor.format());
    EXPECT_EQ(GOLD_DIMS, tensor.dims());
    EXPECT_EQ(GOLD_SHAPE, tensor.shape());
    EXPECT_EQ(GOLD_NDIM, tensor.ndim());
    EXPECT_EQ(GOLD_LAYOUT, tensor.layout());
    ASSERT_NE(nullptr, tensor.handle());

    EXPECT_NE(nullptr, dynamic_cast<nvcv::AllocatorWrapHandle *>(&tensor.alloc()));

    {
        const nvcv::ITensorData *data = tensor.exportData();
        ASSERT_NE(nullptr, data);

        ASSERT_EQ(tensor.format(), data->format());

        auto *devdata = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(data);
        ASSERT_NE(nullptr, devdata);

        EXPECT_EQ(GOLD_NDIM, devdata->ndim());
        ASSERT_EQ(GOLD_DIMS, devdata->dims());
        ASSERT_EQ(GOLD_SHAPE, devdata->shape());
        ASSERT_EQ(GOLD_LAYOUT, devdata->layout());
        ASSERT_EQ(PARAM_FORMAT, devdata->format());

        EXPECT_EQ(devdata->imgPitchBytes(), devdata->pitchBytes(0));
        EXPECT_EQ(devdata->planePitchBytes(), devdata->pitchBytes(1));
        EXPECT_EQ(devdata->numImages(), devdata->shape(0));

        // Write data to each plane
        for (int i = 0; i < devdata->numImages(); ++i)
        {
            for (int p = 0; p < devdata->numPlanes(); ++p)
            {
                void *planeBuffer = devdata->imgPlaneBuffer(i, p);

                ASSERT_EQ(cudaSuccess, cudaMemset2D(planeBuffer, devdata->rowPitchBytes(), i * 3 + p * 7,
                                                    devdata->dims().w * devdata->colPitchBytes(), devdata->dims().h))
                    << "Image #" << i << ", plane #" << p;
            }
        }

        // Check if no overwrites
        for (int i = 0; i < devdata->numImages(); ++i)
        {
            for (int p = 1; p < devdata->numPlanes(); ++p)
            {
                void *planeBuffer = devdata->imgPlaneBuffer(i, p);

                // enough for one plane
                std::vector<uint8_t> buf(devdata->dims().w * devdata->colPitchBytes() * devdata->dims().h);

                ASSERT_EQ(cudaSuccess,
                          cudaMemcpy2D(&buf[0], devdata->dims().w * devdata->colPitchBytes(), planeBuffer,
                                       devdata->rowPitchBytes(), devdata->dims().w * devdata->colPitchBytes(),
                                       devdata->dims().h, cudaMemcpyDeviceToHost))
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

    nvcv::TensorDataPitchDevice buf(fmt, 5, {173, 79}, reinterpret_cast<void *>(678));

    EXPECT_EQ(nvcv::TensorLayout::NCHW, buf.layout());
    EXPECT_EQ(5, buf.dims().n);
    EXPECT_EQ(173, buf.dims().w);
    EXPECT_EQ(79, buf.dims().h);
    EXPECT_EQ(2, buf.dims().c);

    EXPECT_EQ(5, buf.shape()[0]);
    EXPECT_EQ(173, buf.shape()[3]);
    EXPECT_EQ(79, buf.shape()[2]);
    EXPECT_EQ(2, buf.shape()[1]);
    EXPECT_EQ(reinterpret_cast<void *>(678), buf.mem());
    EXPECT_EQ(4, buf.ndim());

    nvcv::TensorWrapData tensor{buf};

    ASSERT_NE(nullptr, tensor.handle());

    EXPECT_NE(nullptr, dynamic_cast<nvcv::AllocatorWrapHandle *>(&tensor.alloc()));

    EXPECT_EQ(buf.dims(), tensor.dims());
    EXPECT_EQ(buf.shape(), tensor.shape());
    EXPECT_EQ(buf.layout(), tensor.layout());
    EXPECT_EQ(buf.ndim(), tensor.ndim());
    EXPECT_EQ(fmt, tensor.format());

    const nvcv::ITensorData *data = tensor.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(data);
    ASSERT_NE(nullptr, devdata);

    EXPECT_EQ(buf.format(), devdata->format());
    EXPECT_EQ(buf.dims(), devdata->dims());
    EXPECT_EQ(buf.shape(), devdata->shape());
    EXPECT_EQ(buf.ndim(), devdata->ndim());

    EXPECT_EQ(buf.mem(), devdata->mem());

    auto *mem = reinterpret_cast<std::byte *>(buf.mem());

    EXPECT_LE(mem + buf.imgPitchBytes() * 4, devdata->imgBuffer(4));
    EXPECT_LE(mem + buf.imgPitchBytes() * 3, devdata->imgBuffer(3));

    EXPECT_LE(mem + buf.imgPitchBytes() * 4, devdata->imgPlaneBuffer(4, 0));
    EXPECT_LE(mem + buf.imgPitchBytes() * 4 + buf.planePitchBytes() * 1, devdata->imgPlaneBuffer(4, 1));

    EXPECT_LE(mem + buf.imgPitchBytes() * 3, devdata->imgPlaneBuffer(3, 0));
    EXPECT_LE(mem + buf.imgPitchBytes() * 3 + buf.planePitchBytes() * 1, devdata->imgPlaneBuffer(3, 1));
}
