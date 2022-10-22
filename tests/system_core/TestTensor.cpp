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
#include <nvcv/Tensor.hpp>

#include <list>
#include <random>

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

    EXPECT_NE(nullptr, dynamic_cast<nvcv::AllocatorWrapHandle *>(&tensor.alloc()));

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
    nvcv::PixelType GOLD_DTYPE = fmt.planePixelType(0);

    nvcv::Tensor::Requirements reqs = nvcv::Tensor::CalcRequirements(5, {173, 79}, fmt);

    nvcv::TensorDataPitchDevice::Buffer buf;
    std::copy(reqs.pitchBytes, reqs.pitchBytes + NVCV_TENSOR_MAX_NDIM, buf.pitchBytes);
    // dummy value, just to check if memory won't be accessed internally. If it does,
    // it'll segfault.
    buf.data = reinterpret_cast<void *>(678);

    nvcv::TensorDataPitchDevice tdata(nvcv::TensorShape{reqs.shape, reqs.ndim, reqs.layout},
                                      nvcv::PixelType{reqs.dtype}, buf);

    EXPECT_EQ(nvcv::TensorLayout::NCHW, tdata.layout());
    EXPECT_EQ(5, tdata.dims().n);
    EXPECT_EQ(173, tdata.dims().w);
    EXPECT_EQ(79, tdata.dims().h);
    EXPECT_EQ(2, tdata.dims().c);

    EXPECT_EQ(5, tdata.shape()[0]);
    EXPECT_EQ(173, tdata.shape()[3]);
    EXPECT_EQ(79, tdata.shape()[2]);
    EXPECT_EQ(2, tdata.shape()[1]);
    EXPECT_EQ(reinterpret_cast<void *>(678), tdata.data());
    EXPECT_EQ(4, tdata.ndim());

    nvcv::TensorWrapData tensor{tdata};

    ASSERT_NE(nullptr, tensor.handle());

    EXPECT_NE(nullptr, dynamic_cast<nvcv::AllocatorWrapHandle *>(&tensor.alloc()));

    EXPECT_EQ(tdata.shape(), tensor.shape());
    EXPECT_EQ(tdata.layout(), tensor.layout());
    EXPECT_EQ(tdata.ndim(), tensor.ndim());
    EXPECT_EQ(GOLD_DTYPE, tensor.dtype());

    const nvcv::ITensorData *data = tensor.exportData();
    ASSERT_NE(nullptr, data);

    auto *devdata = dynamic_cast<const nvcv::ITensorDataPitchDevice *>(data);
    ASSERT_NE(nullptr, devdata);

    EXPECT_EQ(tdata.dtype(), devdata->dtype());
    EXPECT_EQ(tdata.shape(), devdata->shape());
    EXPECT_EQ(tdata.ndim(), devdata->ndim());

    EXPECT_EQ(tdata.data(), devdata->data());

    auto *mem = reinterpret_cast<std::byte *>(tdata.data());

    EXPECT_LE(mem + tdata.imgPitchBytes() * 4, devdata->imgBuffer(4));
    EXPECT_LE(mem + tdata.imgPitchBytes() * 3, devdata->imgBuffer(3));

    EXPECT_LE(mem + tdata.imgPitchBytes() * 4, devdata->imgPlaneBuffer(4, 0));
    EXPECT_LE(mem + tdata.imgPitchBytes() * 4 + tdata.planePitchBytes() * 1, devdata->imgPlaneBuffer(4, 1));

    EXPECT_LE(mem + tdata.imgPitchBytes() * 3, devdata->imgPlaneBuffer(3, 0));
    EXPECT_LE(mem + tdata.imgPitchBytes() * 3 + tdata.planePitchBytes() * 1, devdata->imgPlaneBuffer(3, 1));
}
