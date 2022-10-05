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
#include <nvcv/ColorSpec.h>
#include <util/Compiler.hpp>
#include <util/Size.hpp>

namespace t    = ::testing;
namespace util = nv::cv::util;
namespace test = nv::cv::test;

// Chroma Subsampling ===================================================

TEST(ChromaSubsamplingTests, none_chroma_subsampling_must_be_0)
{
    EXPECT_EQ(0, (int)NVCV_CSS_NONE);
}

class ChromaSubsamplingTests : public t::TestWithParam<std::tuple<NVCVChromaSubsampling, int, int>>
{
};

#define MAKE_CHROMASUB(type, samplesHoriz, samplesVert) \
    {                                                   \
        NVCV_CSS_##type, samplesHoriz, samplesVert      \
    }

NVCV_INSTANTIATE_TEST_SUITE_P(Predefined, ChromaSubsamplingTests,
                              test::ValueList<NVCVChromaSubsampling, int, int>{
                                  MAKE_CHROMASUB(444, 4, 4), MAKE_CHROMASUB(422, 2, 4), MAKE_CHROMASUB(422R, 4, 2),
                                  MAKE_CHROMASUB(411R, 4, 1), MAKE_CHROMASUB(411, 1, 4), MAKE_CHROMASUB(420, 2, 2)});

TEST_P(ChromaSubsamplingTests, predefined_has_correct_definition)
{
    NVCVChromaSubsampling css = std::get<0>(GetParam());

    int goldSamplesHoriz = std::get<1>(GetParam());
    int goldSamplesVert  = std::get<2>(GetParam());

    int samplesHoriz, samplesVert;
    ASSERT_EQ(NVCV_SUCCESS, nvcvChromaSubsamplingGetNumSamples(css, &samplesHoriz, &samplesVert));
    EXPECT_EQ(goldSamplesHoriz, samplesHoriz);
    EXPECT_EQ(goldSamplesVert, samplesVert);
}

TEST_P(ChromaSubsamplingTests, make_chroma_subsampling_function_works)
{
    NVCVChromaSubsampling gold = std::get<0>(GetParam());

    int samplesHoriz = std::get<1>(GetParam());
    int samplesVert  = std::get<2>(GetParam());

    NVCVChromaSubsampling test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvMakeChromaSubsampling(&test, samplesHoriz, samplesVert));

    EXPECT_EQ(gold, test);
}

// Colorspec ===================================================

TEST(ColorSpecTests, get_name_predefined)
{
    EXPECT_STREQ("NVCV_COLOR_SPEC_MPEG2_SMPTE240M", nvcvColorSpecGetName(NVCV_COLOR_SPEC_MPEG2_SMPTE240M));
}

TEST(ColorSpecTests, get_name_non_predefined)
{
    NVCVColorSpec fmt;
    ASSERT_EQ(NVCV_SUCCESS,
              nvcvMakeColorSpec(&fmt, NVCV_COLOR_SPACE_DCIP3, NVCV_YCbCr_ENC_BT2020c, NVCV_COLOR_XFER_sYCC,
                                NVCV_COLOR_RANGE_LIMITED, NVCV_CHROMA_LOC_ODD, NVCV_CHROMA_LOC_CENTER));

    EXPECT_STREQ("NVCVColorSpec(SPACE_DCIP3,ENC_BT2020c,XFER_sYCC,RANGE_LIMITED,LOC_ODD,LOC_CENTER)",
                 nvcvColorSpecGetName(fmt));
}

TEST(ColorSpecTests, set_encoding_to_undefined)
{
    NVCVColorSpec cspec = NVCV_COLOR_SPEC_BT601;
    ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetYCbCrEncoding(&cspec, NVCV_YCbCr_ENC_UNDEFINED));

    EXPECT_EQ(NVCV_COLOR_SPEC_BT601,
              NVCV_MAKE_COLOR_SPEC(NVCV_COLOR_SPACE_BT709, NVCV_YCbCr_ENC_BT601, NVCV_COLOR_XFER_BT709,
                                   NVCV_COLOR_RANGE_LIMITED, NVCV_CHROMA_LOC_EVEN, NVCV_CHROMA_LOC_EVEN));
}

TEST(ColorSpecTests, get_chroma_loc_both_null)
{
    ASSERT_EQ(NVCV_ERROR_INVALID_ARGUMENT, nvcvColorSpecGetChromaLoc(NVCV_COLOR_SPEC_BT601, nullptr, nullptr));
}

class ColorSpecColorRangeTests : public t::TestWithParam<std::tuple<NVCVColorSpec, NVCVColorRange>>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(Full, ColorSpecColorRangeTests,
                              test::ValueList{
                                  NVCV_COLOR_SPEC_UNDEFINED, NVCV_COLOR_SPEC_BT601_ER, NVCV_COLOR_SPEC_BT709_ER,
                                  NVCV_COLOR_SPEC_BT2020_ER, NVCV_COLOR_SPEC_BT2020c_ER, NVCV_COLOR_SPEC_BT2020_PQ_ER,
                                  NVCV_COLOR_SPEC_sRGB, NVCV_COLOR_SPEC_DISPLAYP3_LINEAR, NVCV_COLOR_SPEC_DISPLAYP3,
                                  NVCV_COLOR_SPEC_sYCC, NVCV_COLOR_SPEC_MPEG2_BT601, NVCV_COLOR_SPEC_MPEG2_BT709,
                                  NVCV_COLOR_SPEC_MPEG2_SMPTE240M}
                                  * NVCV_COLOR_RANGE_FULL);

NVCV_INSTANTIATE_TEST_SUITE_P(Limited, ColorSpecColorRangeTests,
                              test::ValueList{NVCV_COLOR_SPEC_BT601, NVCV_COLOR_SPEC_BT709,
                                              NVCV_COLOR_SPEC_BT709_LINEAR, NVCV_COLOR_SPEC_BT2020,
                                              NVCV_COLOR_SPEC_BT2020c, NVCV_COLOR_SPEC_BT2020_PQ,
                                              NVCV_COLOR_SPEC_BT2020_LINEAR, NVCV_COLOR_SPEC_SMPTE240M}
                                  * NVCV_COLOR_RANGE_LIMITED);

TEST_P(ColorSpecColorRangeTests, color_range_correct)
{
    const NVCVColorSpec  cspec = std::get<0>(GetParam());
    const NVCVColorRange gold  = std::get<1>(GetParam());

    NVCVColorRange test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetRange(cspec, &test));
    EXPECT_EQ(gold, test);
}

class ColorSpecColorTransferFunctionTests
    : public t::TestWithParam<std::tuple<NVCVColorSpec, NVCVColorTransferFunction>>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(Linear, ColorSpecColorTransferFunctionTests,
                              test::ValueList{NVCV_COLOR_SPEC_UNDEFINED, NVCV_COLOR_SPEC_BT709_LINEAR,
                                              NVCV_COLOR_SPEC_BT2020_LINEAR, NVCV_COLOR_SPEC_DISPLAYP3_LINEAR}
                                  * NVCV_COLOR_XFER_LINEAR);

NVCV_INSTANTIATE_TEST_SUITE_P(PQ, ColorSpecColorTransferFunctionTests,
                              test::ValueList{NVCV_COLOR_SPEC_BT2020_PQ, NVCV_COLOR_SPEC_BT2020_PQ_ER}
                                  * NVCV_COLOR_XFER_PQ);

NVCV_INSTANTIATE_TEST_SUITE_P(sRGB, ColorSpecColorTransferFunctionTests,
                              test::ValueList{NVCV_COLOR_SPEC_sRGB, NVCV_COLOR_SPEC_DISPLAYP3, NVCV_COLOR_SPEC_sRGB}
                                  * NVCV_COLOR_XFER_sRGB);

NVCV_INSTANTIATE_TEST_SUITE_P(BT601, ColorSpecColorTransferFunctionTests,
                              test::ValueList{NVCV_COLOR_SPEC_BT601, NVCV_COLOR_SPEC_BT601_ER,
                                              NVCV_COLOR_SPEC_MPEG2_BT601, NVCV_COLOR_SPEC_MPEG2_BT709,
                                              NVCV_COLOR_SPEC_BT709, NVCV_COLOR_SPEC_BT709_ER}
                                  * NVCV_COLOR_XFER_BT709);

NVCV_INSTANTIATE_TEST_SUITE_P(BT2020, ColorSpecColorTransferFunctionTests,
                              test::ValueList{NVCV_COLOR_SPEC_BT2020, NVCV_COLOR_SPEC_BT2020c,
                                              NVCV_COLOR_SPEC_BT2020c_ER, NVCV_COLOR_SPEC_BT2020_ER}
                                  * NVCV_COLOR_XFER_BT2020);

NVCV_INSTANTIATE_TEST_SUITE_P(SMPTE240M, ColorSpecColorTransferFunctionTests,
                              test::ValueList{NVCV_COLOR_SPEC_SMPTE240M, NVCV_COLOR_SPEC_MPEG2_SMPTE240M}
                                  * NVCV_COLOR_XFER_SMPTE240M);

NVCV_INSTANTIATE_TEST_SUITE_P(sYCC, ColorSpecColorTransferFunctionTests,
                              test::ValueList{NVCV_COLOR_SPEC_sYCC} * NVCV_COLOR_XFER_sYCC);

TEST_P(ColorSpecColorTransferFunctionTests, color_mapping_correct)
{
    const NVCVColorSpec             cspec = std::get<0>(GetParam());
    const NVCVColorTransferFunction gold  = std::get<1>(GetParam());

    NVCVColorTransferFunction test;
    ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetColorTransferFunction(cspec, &test));
    EXPECT_EQ(gold, test);
}

class ColorModelNeedsColorSpecTests : public t::TestWithParam<std::tuple<NVCVColorModel, bool, NVCVStatus>>
{
};

NVCV_INSTANTIATE_TEST_SUITE_P(Positive, ColorModelNeedsColorSpecTests,
                              test::ValueList<NVCVColorModel, bool>{
                                  {    NVCV_COLOR_MODEL_YCbCr,  true},
                                  {      NVCV_COLOR_MODEL_RGB,  true},
                                  {NVCV_COLOR_MODEL_UNDEFINED, false},
                                  {      NVCV_COLOR_MODEL_RAW, false},
                                  {      NVCV_COLOR_MODEL_XYZ, false},
} * NVCV_SUCCESS);

#if !NVCV_SANITIZED
NVCV_INSTANTIATE_TEST_SUITE_P(Negative, ColorModelNeedsColorSpecTests,
                              test::ValueList<NVCVColorModel, bool>{
                                  {NVCVColorModel(419), true},
} * NVCV_ERROR_INVALID_ARGUMENT);
#endif

TEST_P(ColorModelNeedsColorSpecTests, run)
{
    const NVCVColorModel cmodel     = std::get<0>(GetParam());
    const bool           goldResult = std::get<1>(GetParam());
    const NVCVStatus     goldStatus = std::get<2>(GetParam());

    int8_t testResult = !goldResult;
    ASSERT_EQ(goldStatus, nvcvColorModelNeedsColorspec(cmodel, &testResult));

    if (goldStatus == NVCV_SUCCESS)
    {
        EXPECT_EQ(goldResult, testResult);
    }
    else
    {
        EXPECT_EQ(!goldResult, testResult) << "Must not have changed output";
    }
}

// The tests below explicitly create invalid enums just to test if there's any
// overflow in bitfield representation. This will trigger -fsanitize=enum. Let's
// disable them now in sanitized builds.
#if !NVCV_SANITIZED
TEST(ColorSpecTests, set_color_space)
{
    for (int cspace = 0; cspace < 1 << 3; cspace ? cspace <<= 1 : ++cspace)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(cspace == 0 ? 1 : 0, mask, mask, mask, mask, mask);

        NVCVColorSpec gold = NVCV_MAKE_COLOR_SPEC(cspace, mask, mask, mask, mask, mask);

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetColorSpace(&type, (NVCVColorSpace)cspace));
        EXPECT_EQ(gold, type);
    }
}

TEST(ColorSpecTests, get_color_space)
{
    for (int cspace = 0; cspace < 1 << 3; cspace ? cspace <<= 1 : ++cspace)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(cspace, mask, mask, mask, mask, mask);

        NVCVColorSpace test;
        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetColorSpace(type, &test));
        EXPECT_EQ(cspace, test);
    }
}

TEST(ColorSpecTests, set_encodings)
{
    for (int enc = 0; enc < 1 << 3; enc ? enc <<= 1 : ++enc)
    {
        // One can't set encoding to undefined
        if (enc == NVCV_YCbCr_ENC_UNDEFINED)
        {
            continue;
        }

        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, enc == 0 ? 1 : 0, mask, mask, mask, mask);

        NVCVColorSpec gold = NVCV_MAKE_COLOR_SPEC(mask, enc, mask, mask, mask, mask);

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetYCbCrEncoding(&type, (NVCVYCbCrEncoding)enc));
        ASSERT_EQ(gold, type);
    }
}

TEST(ColorSpecTests, get_encodings)
{
    for (int enc = 0; enc < 1 << 3; enc ? enc <<= 1 : ++enc)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, enc, mask, mask, mask, mask);

        NVCVYCbCrEncoding test;
        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetYCbCrEncoding(type, &test));
        ASSERT_EQ(enc, test);
    }
}

TEST(ColorSpecTests, set_xfer_func)
{
    for (int xfer = 0; xfer < 1 << 3; xfer ? xfer <<= 1 : ++xfer)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, xfer == 0 ? 1 : 0, mask, mask, mask);

        NVCVColorSpec gold = NVCV_MAKE_COLOR_SPEC(mask, mask, xfer, mask, mask, mask);

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetColorTransferFunction(&type, (NVCVColorTransferFunction)xfer));
        ASSERT_EQ(gold, type);
    }
}

TEST(ColorSpecTests, get_xfer_func)
{
    for (int xfer = 0; xfer < 1 << 3; xfer ? xfer <<= 1 : ++xfer)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, xfer, mask, mask, mask);

        NVCVColorTransferFunction test;

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetColorTransferFunction(type, &test));
        ASSERT_EQ(xfer, test);
    }
}

TEST(ColorSpecTests, set_range)
{
    for (int range = 0; range < 1 << 1; range ? range <<= 1 : ++range)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, range == 0 ? 1 : 0, mask, mask);

        NVCVColorSpec gold = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, range, mask, mask);

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetRange(&type, (NVCVColorRange)range));
        ASSERT_EQ(gold, type);
    }
}

TEST(ColorSpecTests, get_range)
{
    for (int range = 0; range < 1 << 1; range ? range <<= 1 : ++range)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, range, mask, mask);

        NVCVColorRange test;
        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetRange(type, &test));
        ASSERT_EQ(range, test);
    }
}

TEST(ColorSpecTests, set_chroma_loc_horiz)
{
    for (int loc = 0; loc < 1 << 2; loc ? loc <<= 1 : ++loc)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, mask, loc == 0 ? 1 : 0, mask);

        NVCVColorSpec gold = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, mask, loc, mask);

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetChromaLoc(&type, (NVCVChromaLocation)loc, (NVCVChromaLocation)mask));
        ASSERT_EQ(gold, type);
    }
}

TEST(ColorSpecTests, get_chroma_loc_horiz)
{
    for (int loc = 0; loc < 1 << 2; loc ? loc <<= 1 : ++loc)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, mask, loc, mask);

        NVCVChromaLocation test;

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetChromaLoc(type, &test, nullptr));
        ASSERT_EQ(loc, test);
    }
}

TEST(ColorSpecTests, set_chroma_loc_vert)
{
    for (int loc = 0; loc < 1 << 2; loc ? loc <<= 1 : ++loc)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, mask, mask, loc == 0 ? 1 : 0);

        NVCVColorSpec gold = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, mask, mask, loc);

        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecSetChromaLoc(&type, (NVCVChromaLocation)mask, (NVCVChromaLocation)loc));
        ASSERT_EQ(gold, type);
    }
}

TEST(ColorSpecTests, get_chroma_loc_vert)
{
    for (int loc = 0; loc < 1 << 2; loc ? loc <<= 1 : ++loc)
    {
        uint64_t mask = UINT64_MAX;

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, mask, mask, loc);

        NVCVChromaLocation test;
        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetChromaLoc(type, nullptr, &test));
        ASSERT_EQ(loc, test);
    }
}

TEST(ColorSpecTests, get_chroma_loc_both)
{
    for (int loc = 0; loc < 1 << 2; loc ? loc <<= 1 : ++loc)
    {
        uint64_t mask = UINT64_MAX;

        NVCVChromaLocation goldHoriz = static_cast<NVCVChromaLocation>(loc & 0b11);
        NVCVChromaLocation goldVert  = static_cast<NVCVChromaLocation>((~loc) & 0b11);

        NVCVColorSpec type = NVCV_MAKE_COLOR_SPEC(mask, mask, mask, mask, goldHoriz, goldVert);

        NVCVChromaLocation testHoriz, testVert;
        ASSERT_EQ(NVCV_SUCCESS, nvcvColorSpecGetChromaLoc(type, &testHoriz, &testVert));
        EXPECT_EQ(goldHoriz, testHoriz);
        EXPECT_EQ(goldVert, testVert);
    }
}
#endif // !NVCV_SANITIZED

// Color Model ===========================

TEST(ColorModelTests, undefined_color_model_is_zero)
{
    EXPECT_EQ(0, (int)NVCV_COLOR_MODEL_UNDEFINED);
}

// YCbCr Encoding ===========================

TEST(ColorModelTests, undefined_ycbcr_encoding_is_zero)
{
    EXPECT_EQ(0, (int)NVCV_YCbCr_ENC_UNDEFINED);
}
