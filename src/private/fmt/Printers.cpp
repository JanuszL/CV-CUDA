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
#include "DataLayout.hpp"

#include <util/Assert.h>

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

std::ostream &operator<<(std::ostream &out, NVCVDataType dataType)
{
    switch (dataType)
    {
        ENUM_CASE(NVCV_DATA_TYPE_UNSIGNED);
        ENUM_CASE(NVCV_DATA_TYPE_SIGNED);
        ENUM_CASE(NVCV_DATA_TYPE_FLOAT);
        ENUM_CASE(NVCV_DATA_TYPE_FLOATISP);
    }
    return out << "NVCVDataType(" << (int)dataType << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVMemLayout memLayout)
{
    switch (memLayout)
    {
        ENUM_CASE(NVCV_MEM_LAYOUT_PITCH_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK1_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK2_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK4_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK8_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK16_LINEAR);
        ENUM_CASE(NVCV_MEM_LAYOUT_BLOCK32_LINEAR);
    }
    return out << "NVCVMemLayout(" << (int)memLayout << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVChannel swizzleChannel)
{
    switch (swizzleChannel)
    {
    case NVCV_CHANNEL_0:
        return out << "0";
    case NVCV_CHANNEL_X:
        return out << "X";
    case NVCV_CHANNEL_Y:
        return out << "Y";
    case NVCV_CHANNEL_Z:
        return out << "Z";
    case NVCV_CHANNEL_W:
        return out << "W";
    case NVCV_CHANNEL_1:
        return out << "1";
    case NVCV_CHANNEL_FORCE8:
        break;
    }
    return out << "NVCVChannel(" << (int)swizzleChannel << ")";
}

std::ostream &operator<<(std::ostream &out, NVCVSwizzle swizzle)
{
    std::array<NVCVChannel, 4> channels = priv::GetChannels(swizzle);

    return out << channels[0] << channels[1] << channels[2] << channels[3];
}

std::ostream &operator<<(std::ostream &out, const NVCVPacking &packing)
{
    if (packing == NVCV_PACKING_0)
    {
        return out << "NVCV_PACKING_0";
    }

    NVCVPackingParams params = priv::GetPackingParams(packing);

    int chbits = 0;
    for (int i = 0; i < 4; ++i)
    {
        if (params.bits[i] != 0)
        {
            chbits += 1;
        }
    }

    if (priv::IsSubWord(params))
    {
        int         nch = priv::GetNumChannels(params.swizzle);
        NVCVChannel ch  = priv::GetSwizzleChannel(params.swizzle, 0);

        if (nch == 1)
        {
            if (ch == NVCV_CHANNEL_X)
            {
                out << ch << params.bits[0] << "b" << params.bits[1];
            }
            else
            {
                out << "b" << params.bits[0] << NVCV_CHANNEL_X << params.bits[1];
            }
        }
        else if (nch == 2)
        {
            if (ch == NVCV_CHANNEL_X)
            {
                out << ch << params.bits[0] << "b" << params.bits[1] << "_" << NVCV_CHANNEL_Y << params.bits[2] << "b"
                    << params.bits[3];
            }
            else
            {
                out << "b" << params.bits[0] << NVCV_CHANNEL_X << params.bits[1] << "_b" << params.bits[2]
                    << NVCV_CHANNEL_Y << params.bits[3];
            }
        }
        else if (nch == 3)
        {
            if (params.swizzle == NVCV_SWIZZLE_YZW0)
            {
                out << "b" << params.bits[0] << NVCV_CHANNEL_X << params.bits[1] << "b" << params.bits[2]
                    << NVCV_CHANNEL_Y << params.bits[3];
            }
            else if (params.swizzle == NVCV_SWIZZLE_XYW0)
            {
                out << NVCV_CHANNEL_X << ch << NVCV_CHANNEL_Y << params.bits[1] << NVCV_CHANNEL_Y << params.bits[2]
                    << NVCV_CHANNEL_Z << params.bits[3];
            }
            else
            {
                NVCV_ASSERT(false);
            }
        }
        else
        {
            NVCV_ASSERT(false);
        }
    }
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            NVCVChannel ch = priv::GetSwizzleChannel(params.swizzle, i);

            if (ch != NVCV_CHANNEL_0)
            {
                if (i == 2)
                {
                    switch (params.swizzle)
                    {
                    case NVCV_SWIZZLE_XYXZ:
                    case NVCV_SWIZZLE_YXZX:
                        out << "_";
                        break;
                    default:
                        break;
                    }
                }

                if (params.endianness != NVCV_HOST_ENDIAN && i > 0)
                {
                    out << "_";
                }

                out << ch << params.bits[i];
            }
        }
    }

    return out;
}

std::ostream &operator<<(std::ostream &out, NVCVEndianness endianness)
{
    switch (endianness)
    {
        ENUM_CASE(NVCV_HOST_ENDIAN);
        ENUM_CASE(NVCV_BIG_ENDIAN);
    }

    return out << "NVCVEndianness(" << (int)endianness << ")";
}
