/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Definitions.hpp"

#include <common/HashUtils.hpp>
#include <common/ValueTests.hpp>
#include <nvcv/TensorDataAccess.hpp>

namespace test = nv::cv::test;
namespace t    = ::testing;
namespace nvcv = nv::cv;

namespace {

class MyTensorDataPitch : public nvcv::ITensorDataPitch
{
public:
    MyTensorDataPitch(nvcv::TensorShape tshape, nvcv::TensorShape::ShapeType pitchBytes, void *mem = nullptr)
        : m_tshape(std::move(tshape))
        , m_mem(mem)
        , m_pitchBytes(std::move(pitchBytes))
    {
        assert((int)pitchBytes.size() == tshape.ndim());

        NVCVTensorData &data = this->cdata();
        data.bufferType      = NVCV_TENSOR_BUFFER_PITCH_DEVICE;
        data.ndim            = tshape.size();
        data.dtype           = NVCV_DATA_TYPE_U8;
        data.layout          = tshape.layout();

        const nvcv::TensorShape::ShapeType &shape = tshape.shape();
        std::copy(shape.begin(), shape.end(), data.shape);

        NVCVTensorBufferPitch &buffer = data.buffer.pitch;
        buffer.data                   = mem;

        std::copy(pitchBytes.begin(), pitchBytes.end(), buffer.pitchBytes);
    }

    bool operator==(const MyTensorDataPitch &that) const
    {
        return std::tie(m_tshape, m_mem, m_pitchBytes) == std::tie(that.m_tshape, that.m_mem, that.m_pitchBytes);
    }

    bool operator<(const MyTensorDataPitch &that) const
    {
        return std::tie(m_tshape, m_mem, m_pitchBytes) < std::tie(that.m_tshape, that.m_mem, that.m_pitchBytes);
    }

    friend void Update(nv::cv::util::HashMD5 &hash, const MyTensorDataPitch &d)
    {
        Update(hash, d.m_tshape, d.m_mem, d.m_pitchBytes);
    }

    friend std::ostream &operator<<(std::ostream &out, const MyTensorDataPitch &d)
    {
        return out << d.m_tshape << ",pitch=" << d.m_pitchBytes << ",mem=" << d.m_mem;
    }

private:
    nvcv::TensorShape            m_tshape;
    void                        *m_mem;
    nvcv::TensorShape::ShapeType m_pitchBytes;
};

} // namespace

// TensorDataAccessPitch::samplePitchBytes ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitch_SamplePitchBytes_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataPitch({{4,34,2},"NxC"},{160,4,2}),160},
        {MyTensorDataPitch({{4,34},"xN"},{10,4}),0},
      });

// clang-format on

TEST_P(TensorDataAccessPitch_SamplePitchBytes_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int64_t           &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessPitch::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->samplePitchBytes());
}

// TensorDataAccessPitch::sampleData ========================

static std::byte *TEST_BASE_ADDR = reinterpret_cast<std::byte *>(0x123);

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitch_SampleData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataPitch({{4,34,2},"NxC"},{160,4,2},TEST_BASE_ADDR),0,TEST_BASE_ADDR+0},
        {MyTensorDataPitch({{4,34,2},"NxC"},{160,4,2},TEST_BASE_ADDR),1,TEST_BASE_ADDR+160},
        {MyTensorDataPitch({{4,34,2},"NxC"},{160,4,2},TEST_BASE_ADDR),2,TEST_BASE_ADDR+2*160},
      });

// clang-format on

TEST_P(TensorDataAccessPitch_SampleData_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int               &idx   = std::get<1>(GetParam());
    void                    *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessPitch::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->sampleData(idx));
}

// TensorDataAccessPitchImage::chPitchBytes ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitchImage_ChannelPitchBytes_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataPitch({{4,34,2},"NWC"},{160,4,2}),2},
        {MyTensorDataPitch({{4,34},"NW"},{160,2}),0},
        {MyTensorDataPitch({{4,34,3,6},"NCHW"},{1042,324,29,12}),324},
      });

// clang-format on

TEST_P(TensorDataAccessPitchImage_ChannelPitchBytes_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int64_t           &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessPitchImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->chPitchBytes());
}

// TensorDataAccessPitchImage::chData ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitch_ChannelData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataPitch({{4,34,3},"NWC"},{160,4,2}, TEST_BASE_ADDR),0, TEST_BASE_ADDR+0},
        {MyTensorDataPitch({{4,34,3},"NWC"},{160,4,2}, TEST_BASE_ADDR),1, TEST_BASE_ADDR+2},
        {MyTensorDataPitch({{4,34,3},"NWC"},{160,4,2}, TEST_BASE_ADDR),2, TEST_BASE_ADDR+4},
      });

// clang-format on

TEST_P(TensorDataAccessPitch_ChannelData_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int               &idx   = std::get<1>(GetParam());
    void                    *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessPitchImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->chData(idx));
}

// TensorDataAccessPitchImage::rowPitchBytes ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitchImage_RowPitchBytes_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataPitch({{4,6,34,2},"NHWC"},{160,32,4,2}),32},
        {MyTensorDataPitch({{4,6,2},"NWC"},{160,32,2}),0},
      });

// clang-format on

TEST_P(TensorDataAccessPitchImage_RowPitchBytes_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int64_t           &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessPitchImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->rowPitchBytes());
}

// TensorDataAccessPitchImage::rowData ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitch_RowData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataPitch({{4,6,34,2},"NHWC"},{160,32,4,2}, TEST_BASE_ADDR), 0, TEST_BASE_ADDR+0},
        {MyTensorDataPitch({{4,6,34,2},"NHWC"},{160,32,4,2}, TEST_BASE_ADDR), 1, TEST_BASE_ADDR+32},
        {MyTensorDataPitch({{4,6,34,2},"NHWC"},{160,32,4,2}, TEST_BASE_ADDR), 2, TEST_BASE_ADDR+64},
      });

// clang-format on

TEST_P(TensorDataAccessPitch_RowData_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int               &idx   = std::get<1>(GetParam());
    void                    *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessPitchImage::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->rowData(idx));
}

// TensorDataAccessPitchImagePlanar::planePitchBytes ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitchImagePlanar_planePitchBytes_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"gold",int64_t>>
      {
        {MyTensorDataPitch({{4,6,34,2},"NHWC"},{160,32,4,2}),0},
        {MyTensorDataPitch({{4,6,2},"NCW"},{160,32,2}),32},
      });

// clang-format on

TEST_P(TensorDataAccessPitchImagePlanar_planePitchBytes_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int64_t           &gold  = std::get<1>(GetParam());

    auto info = nvcv::TensorDataAccessPitchImagePlanar::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->planePitchBytes());
}

// TensorDataAccessPitchImagePlanar::planeData ========================

// clang-format off
NVCV_TEST_SUITE_P(TensorDataAccessPitchImagePlanar_planeData_ExecTests,
      test::ValueList<test::Param<"tdata",MyTensorDataPitch>,
                      test::Param<"idx",int>,
                      test::Param<"gold",void *>>
      {
        {MyTensorDataPitch({{4,6,2},"NCW"},{160,32,2}, TEST_BASE_ADDR),0, TEST_BASE_ADDR+0},
        {MyTensorDataPitch({{4,6,2},"NCW"},{160,32,2}, TEST_BASE_ADDR),1, TEST_BASE_ADDR+32},
        {MyTensorDataPitch({{4,6,2},"NCW"},{160,32,2}, TEST_BASE_ADDR),2, TEST_BASE_ADDR+64},
      });

// clang-format on

TEST_P(TensorDataAccessPitchImagePlanar_planeData_ExecTests, works)
{
    const MyTensorDataPitch &input = std::get<0>(GetParam());
    const int               &idx   = std::get<1>(GetParam());
    void                    *gold  = std::get<2>(GetParam());

    auto info = nvcv::TensorDataAccessPitchImagePlanar::Create(input);
    ASSERT_TRUE(info);
    EXPECT_EQ(gold, info->planeData(idx));
}
