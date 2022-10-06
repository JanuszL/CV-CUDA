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

#ifndef NVCV_PRIV_IMAGE_FORMAT_HPP
#define NVCV_PRIV_IMAGE_FORMAT_HPP

#include "Bitfield.hpp"
#include "ColorSpec.hpp"

#include <core/Exception.hpp>
#include <core/Size.hpp>
#include <nvcv/ImageFormat.h>
#include <nvcv/PixelType.h>
#include <util/StaticVector.hpp>

#include <array>
#include <optional>
#include <string>

namespace nv::cv::priv {

class ColorFormat;
class PixelType;

// Wrapper to NVCVImageFormat to make it properly typed.
class ImageFormat
{
public:
    explicit constexpr ImageFormat(NVCVImageFormat format)
        : m_format(format)
    {
    }

    ImageFormat(NVCVColorModel colorModel, ColorSpec colorSpec, NVCVChromaSubsampling chromaSub,
                NVCVMemLayout memLayout, NVCVDataType dataType, NVCVSwizzle swizzle, NVCVPacking packing0,
                NVCVPacking packing1 = NVCV_PACKING_0, NVCVPacking packing2 = NVCV_PACKING_0,
                NVCVPacking packing3 = NVCV_PACKING_0);

    ImageFormat(NVCVRawPattern rawPattern, NVCVMemLayout memLayout, NVCVDataType dataType, NVCVSwizzle swizzle,
                NVCVPacking packing0, NVCVPacking packing1 = NVCV_PACKING_0, NVCVPacking packing2 = NVCV_PACKING_0,
                NVCVPacking packing3 = NVCV_PACKING_0);

    ImageFormat(NVCVMemLayout memLayout, NVCVDataType dataType, NVCVSwizzle swizzle, NVCVPacking packing0,
                NVCVPacking packing1 = NVCV_PACKING_0, NVCVPacking packing2 = NVCV_PACKING_0,
                NVCVPacking packing3 = NVCV_PACKING_0);

    ImageFormat(const ColorFormat &colorFormat, NVCVChromaSubsampling chromaSub, NVCVMemLayout memLayout,
                NVCVDataType dataType, NVCVSwizzle swizzle, NVCVPacking packing0, NVCVPacking packing1 = NVCV_PACKING_0,
                NVCVPacking packing2 = NVCV_PACKING_0, NVCVPacking packing3 = NVCV_PACKING_0);

    static ImageFormat FromFourCC(uint32_t fourcc, ColorSpec colorSpec, NVCVMemLayout memLayout);

    static ImageFormat FromPlanes(const util::StaticVector<ImageFormat, 4> &planes);

    constexpr NVCVImageFormat value() const noexcept;

    constexpr bool operator==(ImageFormat that) const noexcept;
    constexpr bool operator!=(ImageFormat that) const noexcept;

    ImageFormat            dataType(NVCVDataType newDataType) const;
    constexpr NVCVDataType dataType() const noexcept;

    ImageFormat colorSpec(ColorSpec newColorSpec) const;
    ColorSpec   colorSpec() const noexcept;

    ImageFormat   memLayout(NVCVMemLayout newDataType) const;
    NVCVMemLayout memLayout() const noexcept;

    ImageFormat                   rawPattern(NVCVRawPattern newRawPattern) const;
    std::optional<NVCVRawPattern> rawPattern() const noexcept;

    ImageFormat           css(NVCVChromaSubsampling newCSS) const;
    NVCVChromaSubsampling css() const noexcept;

    ImageFormat    colorRange(NVCVColorRange newColorRange) const;
    NVCVColorRange colorRange() const;

    ImageFormat colorFormat(const ColorFormat &newColorFormat) const;
    ColorFormat colorFormat() const noexcept;

    int                    bpp(int plane) const noexcept;
    NVCVSwizzle            swizzle() const noexcept;
    NVCVColorModel         colorModel() const noexcept;
    int                    blockHeightLog2() const noexcept;
    int                    numChannels() const noexcept;
    std::array<int32_t, 4> bpc() const;
    uint32_t               fourCC() const;
    int                    numPlanes() const noexcept;

    constexpr NVCVPacking planePacking(int plane) const noexcept;
    int                   planePixelStrideBytes(int plane) const noexcept;
    PixelType             planePixelType(int plane) const noexcept;
    int                   planeNumChannels(int plane) const noexcept;
    int                   planeBPP(int plane) const noexcept;
    Size2D                planeSize(Size2D imgSize, int plane) const noexcept;
    NVCVSwizzle           planeSwizzle(int plane) const;
    ImageFormat           planeFormat(int plane) const;

    // Returns whether all planes have same bit depth
    bool hasUniformBitDepth() const;

    // If all planes have the same bit depth, returns it.
    // Or else, returns 0.
    int bitDepth() const;

    ImageFormat swizzleAndPacking(NVCVSwizzle newSwizzle, NVCVPacking newPacking0,
                                  NVCVPacking newPacking1 = NVCV_PACKING_0, NVCVPacking newPacking2 = NVCV_PACKING_0,
                                  NVCVPacking newPacking3 = NVCV_PACKING_0) const;

private:
    NVCVImageFormat m_format;
};

std::ostream &operator<<(std::ostream &out, ImageFormat format);

constexpr NVCVImageFormat ImageFormat::value() const noexcept
{
    return m_format;
}

constexpr bool ImageFormat::operator==(ImageFormat that) const noexcept
{
    return m_format == that.value();
}

constexpr bool ImageFormat::operator!=(ImageFormat that) const noexcept
{
    return !operator==(that);
}

constexpr NVCVPacking ImageFormat::planePacking(int plane) const noexcept
{
    // |11 10 09 08|05 04|03 02 01 00|
    // |  ENC(BPP) |#CH-1|   PACK    |

    auto decode = [](uint32_t plane, uint32_t value, int chlen, int packlen, int bpplen)
    {
        uint32_t bpp  = ExtractBitfield(value, packlen + chlen, bpplen);
        uint32_t nch  = ExtractBitfield(value, packlen, chlen);
        uint32_t pack = ExtractBitfield(value, 0, packlen);

        // if we keep using bpp==0 for 8-bit, since the 4th plane doesn't
        // have a channel count nor packing, the corresponding NVCVPacking
        // would be equal to NVCV_PACKING_0. To avoid it, we don't allow
        // 4th plane to have 128bpp, and we use 128bpp representation (7) to denote 8-bit.
        if (plane == 3 && packlen == 0 && bpp == 0b111)
        {
            // encoding for NVCV_PACKING_X8
            bpp  = 0;
            pack = 4;
        }

        return SetBitfield(bpp, 6, 4) | SetBitfield(nch, 4, 2) | SetBitfield(pack, 0, 4);
    };

    switch (plane)
    {
    case 0:
        return (NVCVPacking)decode(plane, ExtractBitfield(m_format, 35, 9), 2, 3, 4);

    case 1:
        return (NVCVPacking)decode(plane, ExtractBitfield(m_format, 44, 7), 1, 3, 3);

    case 2:
        return (NVCVPacking)decode(plane, ExtractBitfield(m_format, 51, 7), 1, 3, 3);

    case 3:
        return (NVCVPacking)decode(plane, ExtractBitfield(m_format, 58, 3), 0, 0, 3);

    default:
        return NVCV_PACKING_0;
    }
}

constexpr NVCVDataType ImageFormat::dataType() const noexcept
{
    // signed -> unsigned is defined behavior
    return (NVCVDataType)ExtractBitfield(m_format, 61, 3);
}

bool HasSameDataLayout(ImageFormat a, ImageFormat b) noexcept;

// If `cspace` colorspace is undefined, infer its components from 'source'.
ImageFormat UpdateColorSpec(ImageFormat fmt, ColorSpec source);

} // namespace nv::cv::priv

// To be used in a debugger
std::string StrNVCVImageFormat(NVCVImageFormat fmt);

#endif // NVCV_PRIV_IMAGE_FORMAT_HPP
