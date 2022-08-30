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

#include "Printers.hpp"

#include "ColorSpec.hpp"

#include <iostream>

namespace priv = nv::cv::priv;

#define ENUM_CASE(X) \
    case X:          \
        return out << #X

std::ostream &operator<<(std::ostream &out, NVCVColorModel colorModel)
{
    switch (colorModel)
    {
        ENUM_CASE(NVCV_COLOR_MODEL_UNDEFINED);
        ENUM_CASE(NVCV_COLOR_MODEL_RGB);
        ENUM_CASE(NVCV_COLOR_MODEL_YCbCr);
        ENUM_CASE(NVCV_COLOR_MODEL_RAW);
        ENUM_CASE(NVCV_COLOR_MODEL_XYZ);
    }
    return out << "NVCVColorModel(" << (int)colorModel << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVChromaLocation loc)
{
    switch (loc)
    {
        ENUM_CASE(NVCV_CHROMA_LOC_EVEN);
        ENUM_CASE(NVCV_CHROMA_LOC_CENTER);
        ENUM_CASE(NVCV_CHROMA_LOC_ODD);
        ENUM_CASE(NVCV_CHROMA_LOC_BOTH);
    }
    return out << "NVCVChromaLocation(" << (int)loc << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVRawPattern raw)
{
    switch (raw)
    {
        ENUM_CASE(NVCV_RAW_BAYER_RGGB);
        ENUM_CASE(NVCV_RAW_BAYER_BGGR);
        ENUM_CASE(NVCV_RAW_BAYER_GRBG);
        ENUM_CASE(NVCV_RAW_BAYER_GBRG);
        ENUM_CASE(NVCV_RAW_BAYER_RCCB);
        ENUM_CASE(NVCV_RAW_BAYER_BCCR);
        ENUM_CASE(NVCV_RAW_BAYER_CRBC);
        ENUM_CASE(NVCV_RAW_BAYER_CBRC);
        ENUM_CASE(NVCV_RAW_BAYER_RCCC);
        ENUM_CASE(NVCV_RAW_BAYER_CRCC);
        ENUM_CASE(NVCV_RAW_BAYER_CCRC);
        ENUM_CASE(NVCV_RAW_BAYER_CCCR);
        ENUM_CASE(NVCV_RAW_BAYER_CCCC);

    case NVCV_RAW_FORCE8:
        break;
    }

    return out << "NVCVRawPattern(" << (int)raw << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVColorSpace color_space)
{
    switch (color_space)
    {
        ENUM_CASE(NVCV_COLOR_SPACE_BT601);
        ENUM_CASE(NVCV_COLOR_SPACE_BT709);
        ENUM_CASE(NVCV_COLOR_SPACE_BT2020);
        ENUM_CASE(NVCV_COLOR_SPACE_DCIP3);
    }

    return out << "NVCVColorSpace(" << (int)color_space << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVWhitePoint whitePoint)
{
    switch (whitePoint)
    {
        ENUM_CASE(NVCV_WHITE_POINT_D65);
    case NVCV_WHITE_POINT_FORCE8:
        break;
    }

    return out << "NVCVWhitePoint(" << (int)whitePoint << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVColorTransferFunction xferFunc)
{
    switch (xferFunc)
    {
        ENUM_CASE(NVCV_COLOR_XFER_LINEAR);
        ENUM_CASE(NVCV_COLOR_XFER_sRGB);
        ENUM_CASE(NVCV_COLOR_XFER_sYCC);
        ENUM_CASE(NVCV_COLOR_XFER_PQ);
        ENUM_CASE(NVCV_COLOR_XFER_BT709);
        ENUM_CASE(NVCV_COLOR_XFER_BT2020);
        ENUM_CASE(NVCV_COLOR_XFER_SMPTE240M);
    }

    return out << "NVCVColorTransferFunction(" << (int)xferFunc << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVColorRange range)
{
    switch (range)
    {
        ENUM_CASE(NVCV_COLOR_RANGE_FULL);
        ENUM_CASE(NVCV_COLOR_RANGE_LIMITED);
    }

    return out << "NVCVColorRange(" << (int)range << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVYCbCrEncoding encoding)
{
    switch (encoding)
    {
        ENUM_CASE(NVCV_YCbCr_ENC_UNDEFINED);
        ENUM_CASE(NVCV_YCbCr_ENC_BT601);
        ENUM_CASE(NVCV_YCbCr_ENC_BT709);
        ENUM_CASE(NVCV_YCbCr_ENC_BT2020);
        ENUM_CASE(NVCV_YCbCr_ENC_BT2020c);
        ENUM_CASE(NVCV_YCbCr_ENC_SMPTE240M);
    }

    return out << "NVCVYCbCrEncoding(" << (int)encoding << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVChromaSubsampling chromaSub)
{
    switch (chromaSub)
    {
#define ENUM_CASE_CSS(J, a, b, R) \
    case NVCV_CSS_##J##a##b##R:   \
        return out << J << ':' << a << ':' << b << "" #R
        ENUM_CASE_CSS(4, 4, 4, );
        ENUM_CASE_CSS(4, 2, 2, );
        ENUM_CASE_CSS(4, 2, 2, R);
        ENUM_CASE_CSS(4, 1, 1, );
        ENUM_CASE_CSS(4, 1, 1, R);
        ENUM_CASE_CSS(4, 2, 0, );
#undef ENUM_CASE_CSS
    }

    return out << "NVCVChromaSubsampling(" << (int)chromaSub << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVColorSpec cspec)
{
    return out << priv::ColorSpec{cspec};
}
