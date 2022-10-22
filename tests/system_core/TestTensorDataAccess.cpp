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
#include <nvcv/TensorDataAccess.hpp>

namespace test = nv::cv::test;
namespace t    = ::testing;
namespace nvcv = nv::cv;

namespace {

class MyTensorDataPitch : public nvcv::ITensorDataPitch
{
public:
    MyTensorDataPitch(nvcv::TensorShape tshape, nvcv::TensorShape::ShapeType pitchBytes, void *data = nullptr)
        : m_tshape(std::move(tshape))
        , m_data(data)
        , m_pitchBytes(std::move(pitchBytes))
    {
        assert((int)m_pitchBytes.size() == m_tshape.ndim());
    }

    bool operator==(const MyTensorDataPitch &that) const
    {
        return std::tie(m_tshape, m_data, m_pitchBytes) == std::tie(that.m_tshape, that.m_data, that.m_pitchBytes);
    }

    bool operator<(const MyTensorDataPitch &that) const
    {
        return std::tie(m_tshape, m_data, m_pitchBytes) < std::tie(that.m_tshape, that.m_data, that.m_pitchBytes);
    }

    friend void Update(nv::cv::util::HashMD5 &hash, const MyTensorDataPitch &d)
    {
        Update(hash, d.m_tshape, d.m_data, d.m_pitchBytes);
    }

    friend std::ostream &operator<<(std::ostream &out, const MyTensorDataPitch &d)
    {
        return out << d.m_tshape << ",pitch=" << d.m_pitchBytes << ",data=" << d.m_data;
    }

private:
    nvcv::TensorShape            m_tshape;
    void                        *m_data;
    nvcv::TensorShape::ShapeType m_pitchBytes;

    int doGetNumDim() const override
    {
        return m_tshape.ndim();
    }

    const nvcv::TensorShape &doGetShape() const override
    {
        return m_tshape;
    }

    const nvcv::TensorShape::DimType &doGetShapeDim(int d) const override
    {
        return m_tshape[d];
    }

    nvcv::PixelType doGetPixelType() const override
    {
        return nvcv::TYPE_U8;
    }

    void *doGetData() const override
    {
        return m_data;
    }

    const int64_t &doGetPitchBytes(int d) const override
    {
        assert(0 <= d && d < m_pitchBytes.size());
        return m_pitchBytes[d];
    }

    nvcv::DimsNCHW doGetDims() const override
    {
        assert(false && !"should not be needed");
        return {};
    }

    int32_t doGetNumPlanes() const override
    {
        assert(false && !"should not be needed");
        return 0;
    }

    int32_t doGetNumImages() const override
    {
        assert(false && !"should not be needed");
        return 0;
    }

    const NVCVTensorData &doGetCData() const override
    {
        assert(false && !"should not be needed");
        static NVCVTensorData tdata;
        return tdata;
    }

    int64_t doGetImagePitchBytes() const override
    {
        assert(false && !"should not be needed");
        return 0;
    }

    int64_t doGetPlanePitchBytes() const override
    {
        assert(false && !"should not be needed");
        return 0;
    }

    int64_t doGetRowPitchBytes() const override
    {
        assert(false && !"should not be needed");
        return 0;
    }

    int64_t doGetColPitchBytes() const override
    {
        assert(false && !"should not be needed");
        return 0;
    }

    void *doGetImageBuffer(int n) const override
    {
        assert(false && !"should not be needed");
        return 0;
    }

    void *doGetImagePlaneBuffer(int n, int p) const override
    {
        assert(false && !"should not be needed");
        return 0;
    }
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
