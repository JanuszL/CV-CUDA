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

#include "ColorSpec.hpp"

#include "Bitfield.hpp"
#include "Printers.hpp"

#include <core/Exception.hpp>

#include <sstream>

namespace nv::cv::priv {

ColorSpec::ColorSpec(NVCVColorSpace cspace, NVCVYCbCrEncoding encoding, NVCVColorTransferFunction xferfunc,
                     NVCVColorRange range, const ChromaLoc &loc) noexcept
    : m_cspec{NVCV_MAKE_COLOR_SPEC(cspace, encoding, xferfunc, range, loc.horiz, loc.vert)}
{
}

ColorSpec::operator NVCVColorSpec() const noexcept
{
    return m_cspec;
}

NVCVColorSpace ColorSpec::colorSpace() const noexcept
{
    int32_t val = ExtractBitfield(m_cspec, 0, 3);
    return (NVCVColorSpace)val;
}

ColorSpec ColorSpec::colorSpace(NVCVColorSpace newColorSpace) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(0, 3)) | SetBitfield(newColorSpace, 0, 3))};
}

ColorSpec ColorSpec::YCbCrEncoding(NVCVYCbCrEncoding newEncoding) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(7, 3)) | SetBitfield(newEncoding, 7, 3))};
}

NVCVYCbCrEncoding ColorSpec::YCbCrEncoding() const noexcept
{
    return (NVCVYCbCrEncoding)ExtractBitfield(m_cspec, 7, 3);
}

ColorSpec ColorSpec::xferFunc(NVCVColorTransferFunction newXferFunc) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(3, 4)) | SetBitfield(newXferFunc, 3, 4))};
}

NVCVColorTransferFunction ColorSpec::xferFunc() const noexcept
{
    return (NVCVColorTransferFunction)ExtractBitfield(m_cspec, 3, 4);
}

ColorSpec ColorSpec::colorRange(NVCVColorRange newRange) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(14, 1)) | SetBitfield(newRange, 14, 1))};
}

NVCVColorRange ColorSpec::colorRange() const noexcept
{
    return (NVCVColorRange)ExtractBitfield(m_cspec, 14, 1);
}

ChromaLoc ColorSpec::chromaLoc() const noexcept
{
    return {
        (NVCVChromaLocation)ExtractBitfield(m_cspec, 10, 2), // horiz
        (NVCVChromaLocation)ExtractBitfield(m_cspec, 12, 2)  // vert
    };
}

ColorSpec ColorSpec::chromaLoc(const ChromaLoc &loc) const
{
    return ColorSpec{(NVCVColorSpec)(((uint64_t)m_cspec & ~MaskBitfield(10, 4)) | SetBitfield(loc.horiz, 10, 2)
                                     | SetBitfield(loc.vert, 12, 2))};
}

NVCVWhitePoint ColorSpec::whitePoint() const noexcept
{
    // so far we only support D65...
    return NVCV_WHITE_POINT_D65;
}

NVCVChromaSubsampling MakeNVCVChromaSubsampling(int samplesHoriz, int samplesVert)
{
    switch (samplesHoriz)
    {
    case 4:
        switch (samplesVert)
        {
        case 4:
            return NVCV_CSS_444;
        case 2:
            return NVCV_CSS_422R;
        case 1:
            return NVCV_CSS_411R;
        }
        break;

    case 2:
        switch (samplesVert)
        {
        case 4:
            return NVCV_CSS_422;
        case 2:
            return NVCV_CSS_420;
        }
        break;

    case 1:
        switch (samplesVert)
        {
        case 1:
            return NVCV_CSS_444;
        case 4:
            return NVCV_CSS_411;
        }
        break;
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
        << samplesHoriz << " horizontal and " << samplesVert
        << " vertical samples doesn't correspond to any supported chroma subsampling scheme";
}

std::pair<int, int> GetChromaSamples(NVCVChromaSubsampling css)
{
    switch (css)
    {
    case NVCV_CSS_444:
        return {4, 4};

    case NVCV_CSS_422R:
        return {4, 2};

    case NVCV_CSS_411R:
        return {4, 1};

    case NVCV_CSS_422:
        return {2, 4};

    case NVCV_CSS_420:
        return {2, 2};

    case NVCV_CSS_411:
        return {1, 4};
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid chroma subsampling: " << css;
}

std::ostream &operator<<(std::ostream &out, ColorSpec cspec)
{
    switch (cspec)
    {
#define ENUM_CASE(X) \
    case X:          \
        return out << #X
        ENUM_CASE(NVCV_COLOR_SPEC_UNDEFINED);
        ENUM_CASE(NVCV_COLOR_SPEC_MPEG2_BT601);
        ENUM_CASE(NVCV_COLOR_SPEC_MPEG2_BT709);
        ENUM_CASE(NVCV_COLOR_SPEC_MPEG2_SMPTE240M);
        ENUM_CASE(NVCV_COLOR_SPEC_BT601);
        ENUM_CASE(NVCV_COLOR_SPEC_BT601_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT709);
        ENUM_CASE(NVCV_COLOR_SPEC_BT709_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT709_LINEAR);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_LINEAR);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_PQ);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020_PQ_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020c);
        ENUM_CASE(NVCV_COLOR_SPEC_BT2020c_ER);
        ENUM_CASE(NVCV_COLOR_SPEC_SMPTE240M);
        ENUM_CASE(NVCV_COLOR_SPEC_sRGB);
        ENUM_CASE(NVCV_COLOR_SPEC_sYCC);
        ENUM_CASE(NVCV_COLOR_SPEC_DISPLAYP3);
        ENUM_CASE(NVCV_COLOR_SPEC_DISPLAYP3_LINEAR);
#undef ENUM_CASE
    case NVCV_COLOR_SPEC_FORCE32:
        out << "NVCVColorSpec(invalid)";
        break;
    }

    out << "NVCVColorSpec(" << cspec.colorSpace() << "," << cspec.YCbCrEncoding() << "," << cspec.xferFunc() << ","
        << cspec.colorRange() << "," << cspec.chromaLoc().horiz << "," << cspec.chromaLoc().vert << ")";
    return out;
}

bool NeedsColorspec(NVCVColorModel cmodel)
{
    switch (cmodel)
    {
    case NVCV_COLOR_MODEL_YCbCr:
    case NVCV_COLOR_MODEL_RGB:
        return true;
    case NVCV_COLOR_MODEL_UNDEFINED:
    case NVCV_COLOR_MODEL_RAW:
    case NVCV_COLOR_MODEL_XYZ:
        return false;
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid color model: " << cmodel;
}

} // namespace nv::cv::priv

std::string StrNVCVColorSpec(NVCVColorSpec cspec)
{
    std::ostringstream ss;
    ss << nv::cv::priv::ColorSpec{cspec};
    return ss.str();
}
