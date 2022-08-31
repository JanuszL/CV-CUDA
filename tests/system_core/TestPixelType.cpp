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
#include <nvcv/ImageFormat.hpp>
#include <nvcv/PixelType.h>
#include <nvcv/PixelType.hpp>

namespace t    = ::testing;
namespace util = nv::cv::util;
namespace test = nv::cv::test;

namespace {

struct Params
{
    NVCVPixelType pixType;
    NVCVMemLayout memLayout;
    NVCVPacking   packing;
    NVCVDataType  dataType;
    int           channels;
    int           bpp;
};

std::ostream &operator<<(std::ostream &out, const Params &p)
{
    return out << "pixType=" << nvcvPixelTypeGetName(p.pixType) << ", memLayout=" << p.memLayout
               << ", packing=" << p.packing << ", dataType=" << p.dataType;
}

} // namespace

class PixelTypeTests : public t::TestWithParam<Params>
{
};

#define FMT_PIXEL_PARAMS(MemLayout, DataType, Packing) \
    NVCV_MEM_LAYOUT_##MemLayout, NVCV_PACKING_##Packing, NVCV_DATA_TYPE_##DataType

#define MAKE_PIXEL_TYPE(MemLayout, DataType, Packing)                                                     \
    NVCV_MAKE_PIXEL_TYPE(NVCV_MEM_LAYOUT_##MemLayout, NVCV_DATA_TYPE_##DataType, NVCV_PACKING_##Packing), \
        FMT_PIXEL_PARAMS(MemLayout, DataType, Packing)

INSTANTIATE_TEST_SUITE_P(Range, PixelTypeTests, t::Values(Params{MAKE_PIXEL_TYPE(BL, FLOAT, X32_Y32_Z32_W32), 4, 128}));

INSTANTIATE_TEST_SUITE_P(ExplicitTypes, PixelTypeTests,
                         t::Values(Params{NVCV_PIXEL_TYPE_U8, FMT_PIXEL_PARAMS(PL, UNSIGNED, X8), 1, 8},
                                   Params{NVCV_PIXEL_TYPE_S8, FMT_PIXEL_PARAMS(PL, SIGNED, X8), 1, 8},
                                   Params{NVCV_PIXEL_TYPE_U16, FMT_PIXEL_PARAMS(PL, UNSIGNED, X16), 1, 16},
                                   Params{NVCV_PIXEL_TYPE_S16, FMT_PIXEL_PARAMS(PL, SIGNED, X16), 1, 16},
                                   Params{NVCV_PIXEL_TYPE_2U8, FMT_PIXEL_PARAMS(PL, UNSIGNED, X8_Y8), 2, 16},
                                   Params{NVCV_PIXEL_TYPE_3U8, FMT_PIXEL_PARAMS(PL, UNSIGNED, X8_Y8_Z8), 3, 24},
                                   Params{NVCV_PIXEL_TYPE_4U8, FMT_PIXEL_PARAMS(PL, UNSIGNED, X8_Y8_Z8_W8), 4, 32},
                                   Params{NVCV_PIXEL_TYPE_F32, FMT_PIXEL_PARAMS(PL, FLOAT, X32), 1, 32},
                                   Params{NVCV_PIXEL_TYPE_F64, FMT_PIXEL_PARAMS(PL, FLOAT, X64), 1, 64},
                                   Params{MAKE_PIXEL_TYPE(PL, UNSIGNED, X10Y11Z11), 3, 32},
                                   Params{MAKE_PIXEL_TYPE(PL, UNSIGNED, X5Y6Z5), 3, 16},
                                   Params{MAKE_PIXEL_TYPE(PL, UNSIGNED, X10b6_Y10b6), 2, 32},
                                   Params{NVCV_PIXEL_TYPE_2F32, FMT_PIXEL_PARAMS(PL, FLOAT, X32_Y32), 2, 64}));

TEST_P(PixelTypeTests, get_comp_packing_works)
{
    const Params &p = GetParam();

    NVCVPacking packing;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetPacking(p.pixType, &packing));
    EXPECT_EQ(p.packing, packing);
}

TEST_P(PixelTypeTests, get_bpp_works)
{
    const Params &p = GetParam();

    int32_t bpp;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetBitsPerPixel(p.pixType, &bpp));
    EXPECT_EQ(p.bpp, bpp);
}

TEST_P(PixelTypeTests, get_bpc_works)
{
    const Params &p = GetParam();

    int32_t bits[4];
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetBitsPerChannel(p.pixType, bits));

    NVCVPacking packing;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetPacking(p.pixType, &packing));

    int32_t goldbits[4];
    ASSERT_EQ(NVCV_SUCCESS, nvcvPackingGetBitsPerComponent(packing, goldbits));

    EXPECT_EQ(bits[0], goldbits[0]);
    EXPECT_EQ(bits[1], goldbits[1]);
    EXPECT_EQ(bits[2], goldbits[2]);
    EXPECT_EQ(bits[3], goldbits[3]);
}

TEST_P(PixelTypeTests, get_datatype_works)
{
    const Params &p = GetParam();

    NVCVDataType dataType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetDataType(p.pixType, &dataType));
    EXPECT_EQ(p.dataType, dataType);
}

TEST_P(PixelTypeTests, get_memlayout_works)
{
    const Params &p = GetParam();

    NVCVMemLayout memLayout;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetMemLayout(p.pixType, &memLayout));
    EXPECT_EQ(p.memLayout, memLayout);
}

TEST_P(PixelTypeTests, set_memlayout_works)
{
    const Params &p = GetParam();

    NVCVMemLayout newLayout = p.memLayout == NVCV_MEM_LAYOUT_PL ? NVCV_MEM_LAYOUT_BL : NVCV_MEM_LAYOUT_PL;

    NVCVPixelType pix = p.pixType;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeSetMemLayout(&pix, newLayout));

    NVCVMemLayout layout;
    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetMemLayout(pix, &layout));
    EXPECT_EQ(newLayout, layout);
}

TEST_P(PixelTypeTests, get_channel_count)
{
    const Params &p = GetParam();

    int channels;

    ASSERT_EQ(NVCV_SUCCESS, nvcvPixelTypeGetNumChannels(p.pixType, &channels));
    EXPECT_EQ(p.channels, channels);
}

TEST_P(PixelTypeTests, make_pixel_type)
{
    const Params &p = GetParam();

    NVCVPixelType pix;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakePixelType(&pix, p.memLayout, p.dataType, p.packing));
    EXPECT_EQ(p.pixType, pix);
}

class ImagePixelTypeTests : public t::TestWithParam<std::tuple<nv::cv::ImageFormat, nv::cv::PixelType>>
{
};

INSTANTIATE_TEST_SUITE_P(ExplicitTypes, ImagePixelTypeTests,
                         t::Values(std::make_tuple(NVCV_IMAGE_FORMAT_U8, NVCV_PIXEL_TYPE_U8),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_S8, NVCV_PIXEL_TYPE_S8),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_U16, NVCV_PIXEL_TYPE_U16),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_S16, NVCV_PIXEL_TYPE_S16),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_F32, NVCV_PIXEL_TYPE_F32),
                                   std::make_tuple(NVCV_IMAGE_FORMAT_2F32, NVCV_PIXEL_TYPE_2F32)));

TEST_P(ImagePixelTypeTests, pixel_type_matches_corresponding_image_type)
{
    NVCVImageFormat imgFormat = std::get<0>(GetParam());
    NVCVPixelType   pixType   = std::get<1>(GetParam());

    EXPECT_EQ((uint64_t)imgFormat, (uint64_t)pixType);
}

TEST(PixelTypeTests, pixel_type_none_must_be_0)
{
    EXPECT_EQ(0, (int)NVCV_PIXEL_TYPE_NONE);
}

TEST(PixelTypeTests, make_pixel_type_macro)
{
    NVCVPixelType pixType;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakePixelType(&pixType, NVCV_MEM_LAYOUT_PL, NVCV_DATA_TYPE_UNSIGNED, NVCV_PACKING_X8_Y8_Z8));

    EXPECT_EQ(pixType, NVCV_MAKE_PIXEL_TYPE(NVCV_MEM_LAYOUT_PL, NVCV_DATA_TYPE_UNSIGNED, NVCV_PACKING_X8_Y8_Z8));
}

TEST(PixelTypeTests, get_name_predefined)
{
    EXPECT_STREQ("NVCV_PIXEL_TYPE_U8", nvcvPixelTypeGetName(NVCV_PIXEL_TYPE_U8));
}

TEST(PixelTypeTests, get_name_not_predefined)
{
    NVCVPixelType pix
        = NVCV_MAKE_PIXEL_TYPE(NVCV_MEM_LAYOUT_BLOCK2_LINEAR, NVCV_DATA_TYPE_FLOAT, NVCV_PACKING_X8_Y8_Z8);

    EXPECT_STREQ("NVCVPixelType(BLOCK2_LINEAR,FLOAT,X8_Y8_Z8)", nvcvPixelTypeGetName(pix));
}

class ChannelPixelTypeTests : public t::TestWithParam<std::tuple<NVCVPixelType, int, NVCVPixelType, NVCVStatus>>
{
};

#define MAKE_PIXEL_TYPE_ABBREV(layout, datatype, packing) \
    NVCV_MAKE_PIXEL_TYPE(NVCV_MEM_LAYOUT_##layout, NVCV_DATA_TYPE_##datatype, NVCV_PACKING_##packing)

// clang-format off
NVCV_INSTANTIATE_TEST_SUITE_P(
    Positive, ChannelPixelTypeTests,
    test::ValueList<NVCVPixelType, int, NVCVPixelType>
    {
        {NVCV_PIXEL_TYPE_U8,                          0, NVCV_PIXEL_TYPE_U8},
        {MAKE_PIXEL_TYPE_ABBREV(BL, FLOAT, X32),      0, MAKE_PIXEL_TYPE_ABBREV(BL, FLOAT, X32)},
        {MAKE_PIXEL_TYPE_ABBREV(BL, FLOAT, X8_Y8_Z8), 2, MAKE_PIXEL_TYPE_ABBREV(BL, FLOAT, X8)},
        {NVCV_PIXEL_TYPE_2F32,                        0, NVCV_PIXEL_TYPE_F32},
        {NVCV_PIXEL_TYPE_2F32,                        1, NVCV_PIXEL_TYPE_F32},
        {MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X64_Y64_Z64_W64), 3, NVCV_PIXEL_TYPE_S64},
        {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X4Y4),  0, MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X4)},
        {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, b4X4Y4Z4), 2, MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X4)},
        {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X3Y3Z2), 2, MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X2)},
        {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X1),    0, MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X1)},
        {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X2),    0, MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X2)},
        {NVCV_PIXEL_TYPE_3S16,                        2, NVCV_PIXEL_TYPE_S16},
        {MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X24),     0, MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X24)},
        {NVCV_PIXEL_TYPE_2F64,                        1, NVCV_PIXEL_TYPE_F64},
        {MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X96),     0, MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X96)},
        {MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X128),    0, MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X128)},
        {MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X192),    0, MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X192)},
        {MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X256),    0, MAKE_PIXEL_TYPE_ABBREV(PL, SIGNED, X256)},
        {NVCV_PIXEL_TYPE_U8,                          1, NVCV_PIXEL_TYPE_NONE},
        {NVCV_PIXEL_TYPE_2F32,                        2, NVCV_PIXEL_TYPE_NONE},
        {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, b4X4Y4Z4), 3, NVCV_PIXEL_TYPE_NONE},
    }
        * NVCV_SUCCESS);

NVCV_INSTANTIATE_TEST_SUITE_P(
    Negative, ChannelPixelTypeTests,
    test::ValueList<NVCVPixelType, int>
    {
          {NVCV_PIXEL_TYPE_U8, -1},
          {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, b2X14), 0},
          {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X5Y5b1Z5), 0},
          {MAKE_PIXEL_TYPE_ABBREV(PL, UNSIGNED, X3Y3Z2), 1}
    }
    * NVCV_PIXEL_TYPE_NONE * NVCV_ERROR_INVALID_ARGUMENT);

// clang-format on

TEST_P(ChannelPixelTypeTests, get_channel_type)
{
    NVCVPixelType test       = std::get<0>(GetParam());
    int           channel    = std::get<1>(GetParam());
    NVCVPixelType gold       = std::get<2>(GetParam());
    NVCVStatus    goldStatus = std::get<3>(GetParam());

    NVCVPixelType pix;
    ASSERT_EQ(goldStatus, nvcvPixelTypeGetChannelType(test, channel, &pix));
    if (goldStatus == NVCV_SUCCESS)
    {
        EXPECT_EQ(gold, pix);
    }
}
