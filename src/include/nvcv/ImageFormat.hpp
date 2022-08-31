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

/**
 * @file ImageFormat.hpp
 *
 * @brief Defines C++ types and functions to handle image formats.
 */

#ifndef NVCV_IMAGE_FORMAT_HPP
#define NVCV_IMAGE_FORMAT_HPP

#include "ColorSpec.hpp"
#include "DataLayout.hpp"
#include "ImageFormat.h"
#include "PixelType.hpp"
#include "Size.hpp"

#include <array>
#include <iostream>

namespace nv::cv {

class ImageFormat
{
public:
    constexpr ImageFormat();
    explicit constexpr ImageFormat(NVCVImageFormat format);

    ImageFormat(ColorSpec colorSpec, ChromaSubsampling chromaSub, MemLayout memLayout, DataType dataType,
                Swizzle swizzle, Packing packing0, Packing packing1 = Packing::NONE, Packing packing2 = Packing::NONE,
                Packing packing3 = Packing::NONE);

    ImageFormat(ColorModel colorModel, ColorSpec colorSpec, MemLayout memLayout, DataType dataType, Swizzle swizzle,
                Packing packing0, Packing packing1 = Packing::NONE, Packing packing2 = Packing::NONE,
                Packing packing3 = Packing::NONE);

    ImageFormat(MemLayout memLayout, DataType dataType, Swizzle swizzle, Packing packing0,
                Packing packing1 = Packing::NONE, Packing packing2 = Packing::NONE, Packing packing3 = Packing::NONE);

    ImageFormat(RawPattern rawPattern, MemLayout memLayout, DataType dataType, Swizzle swizzle, Packing packing0,
                Packing packing1 = Packing::NONE, Packing packing2 = Packing::NONE, Packing packing3 = Packing::NONE);

    static constexpr ImageFormat ConstCreate(ColorSpec colorSpec, ChromaSubsampling chromaSub, MemLayout memLayout,
                                             DataType dataType, Swizzle swizzle, Packing packing0,
                                             Packing packing1 = Packing::NONE, Packing packing2 = Packing::NONE,
                                             Packing packing3 = Packing::NONE);

    static constexpr ImageFormat ConstCreate(ColorModel colorModel, ColorSpec colorSpec, MemLayout memLayout,
                                             DataType dataType, Swizzle swizzle, Packing packing0,
                                             Packing packing1 = Packing::NONE, Packing packing2 = Packing::NONE,
                                             Packing packing3 = Packing::NONE);

    static constexpr ImageFormat ConstCreate(MemLayout memLayout, DataType dataType, Swizzle swizzle, Packing packing0,
                                             Packing packing1 = Packing::NONE, Packing packing2 = Packing::NONE,
                                             Packing packing3 = Packing::NONE);

    static constexpr ImageFormat ConstCreate(RawPattern rawPattern, MemLayout memLayout, DataType dataType,
                                             Swizzle swizzle, Packing packing0, Packing packing1 = Packing::NONE,
                                             Packing packing2 = Packing::NONE, Packing packing3 = Packing::NONE);

    static ImageFormat FromFourCC(uint fourcc, ColorSpec colorSpec, MemLayout memLayout);

    static ImageFormat FromPlanes(ImageFormat plane0, ImageFormat plane1 = {}, ImageFormat plane2 = {},
                                  ImageFormat plane3 = {});

    constexpr                 operator NVCVImageFormat() const noexcept;
    constexpr NVCVImageFormat cvalue() const noexcept;

    constexpr bool operator==(ImageFormat that) const noexcept;
    constexpr bool operator!=(ImageFormat that) const noexcept;

    ImageFormat dataType(DataType dataType) const;
    DataType    dataType() const noexcept;

    ImageFormat memLayout(MemLayout newMemLayout) const;
    MemLayout   memLayout() const noexcept;

    ImageFormat colorSpec(ColorSpec newColorSpec) const;
    ColorSpec   colorSpec() const noexcept;

    ImageFormat       chromaSubsampling(ChromaSubsampling css) const;
    ChromaSubsampling chromaSubsampling() const noexcept;

    ImageFormat rawPattern(RawPattern newRawPattern) const;
    RawPattern  rawPattern() const noexcept;

    int                bitsPerPixel(int plane) const noexcept;
    Swizzle            swizzle() const noexcept;
    ColorModel         colorModel() const noexcept;
    int                blockHeightLog2() const noexcept;
    int                numChannels() const noexcept;
    std::array<int, 4> bitsPerChannel() const noexcept;
    uint               fourCC() const;
    ColorRange         colorRange() const;
    int                bitDepth() const;
    bool               hasUniformBitDepth() const;
    int                numPlanes() const noexcept;

    ImageFormat swizzleAndPacking(Swizzle newSwizzle, Packing newPacking0, Packing newPacking1, Packing newPacking2,
                                  Packing newPacking3) const;

    Packing     planePacking(int plane) const noexcept;
    int         planeElemStride(int plane) const noexcept;
    PixelType   planePixelType(int plane) const noexcept;
    int         planeNumChannels(int plane) const noexcept;
    int         planeBitsPerPixel(int plane) const noexcept;
    Size2D      planeSize(Size2D imgSize, int plane) const noexcept;
    Swizzle     planeSwizzle(int plane) const noexcept;
    ImageFormat planeFormat(int plane) const noexcept;

private:
    NVCVImageFormat m_format;
};

constexpr ImageFormat::ImageFormat()
    : m_format(NVCV_IMAGE_FORMAT_NONE)
{
}

constexpr ImageFormat::ImageFormat(NVCVImageFormat format)
    : m_format(format)
{
}

/** Single plane with one 8-bit unsigned integer channel. */
constexpr ImageFormat FMT_U8{NVCV_IMAGE_FORMAT_U8};

/** Single plane with one block-linear 8-bit unsigned integer channel. */
constexpr ImageFormat FMT_U8_BL{NVCV_IMAGE_FORMAT_U8_BL};

/** Single plane with one 8-bit signed integer channel. */
constexpr ImageFormat FMT_S8{NVCV_IMAGE_FORMAT_S8};

/** Single plane with one 16-bit unsigned integer channel. */
constexpr ImageFormat FMT_U16{NVCV_IMAGE_FORMAT_U16};

/** Single plane with one 32-bit unsigned integer channel. */
constexpr ImageFormat FMT_U32{NVCV_IMAGE_FORMAT_U32};

/** Single plane with one 32-bit signed integer channel.*/
constexpr ImageFormat FMT_S32{NVCV_IMAGE_FORMAT_S32};

/** Single plane with one 16-bit signed integer channel.*/
constexpr ImageFormat FMT_S16{NVCV_IMAGE_FORMAT_S16};

/** Single plane with one block-linear 16-bit signed integer channel.*/
constexpr ImageFormat FMT_S16_BL{NVCV_IMAGE_FORMAT_S16_BL};

/** Single plane with two interleaved 16-bit signed integer channel.*/
constexpr ImageFormat FMT_2S16{NVCV_IMAGE_FORMAT_2S16};

/** Single plane with two interleaved block-linear 16-bit signed integer channel.*/
constexpr ImageFormat FMT_2S16_BL{NVCV_IMAGE_FORMAT_2S16_BL};

/** Single plane with one 32-bit floating point channel. */
constexpr ImageFormat FMT_F32{NVCV_IMAGE_FORMAT_F32};

/** Single plane with one 64-bit floating point channel. */
constexpr ImageFormat FMT_F64{NVCV_IMAGE_FORMAT_F64};

/** Single plane with two interleaved 32-bit floating point channels. */
constexpr ImageFormat FMT_2F32{NVCV_IMAGE_FORMAT_2F32};

/** Single plane with one pitch-linear 8-bit unsigned integer channel with limited-range luma (grayscale) information.
     *  Values range from 16 to 235. Below this range is considered black, above is considered white.
     */
constexpr ImageFormat FMT_Y8{NVCV_IMAGE_FORMAT_Y8};

/** Single plane with one block-linear 8-bit unsigned integer channel with limited-range luma (grayscale) information.
     *  Values range from 16 to 235. Below this range is considered black, above is considered white.
     */
constexpr ImageFormat FMT_Y8_BL{NVCV_IMAGE_FORMAT_Y8_BL};

/** Single plane with one pitch-linear 8-bit unsigned integer channel with full-range luma (grayscale) information.
     *  Values range from 0 to 255.
     */
constexpr ImageFormat FMT_Y8_ER{NVCV_IMAGE_FORMAT_Y8_ER};

/** Single plane with one block-linear 8-bit unsigned integer channel with full-range luma (grayscale) information.
     *  Values range from 0 to 255.
     */
constexpr ImageFormat FMT_Y8_ER_BL{NVCV_IMAGE_FORMAT_Y8_ER_BL};

/** Single plane with one pitch-linear 16-bit unsigned integer channel with limited-range luma (grayscale) information.
     *  Values range from 4096 to 60160. Below this range is considered black, above is considered white.
     */
constexpr ImageFormat FMT_Y16{NVCV_IMAGE_FORMAT_Y16};

/** Single plane with one block-linear 16-bit unsigned integer channel with limited-range luma (grayscale) information.
     *  Values range from 4096 to 60160. Below this range is considered black, above is considered white.
     */
constexpr ImageFormat FMT_Y16_BL{NVCV_IMAGE_FORMAT_Y16_BL};

/** Single plane with one pitch-linear 16-bit unsigned integer channel with full-range luma (grayscale) information.
     *  Values range from 0 to 65535.
     */
constexpr ImageFormat FMT_Y16_ER{NVCV_IMAGE_FORMAT_Y16_ER};

/** Single plane with one block-linear 16-bit unsigned integer channel with full-range luma (grayscale) information.
     *  Values range from 0 to 65535.
     */
constexpr ImageFormat FMT_Y16_ER_BL{NVCV_IMAGE_FORMAT_Y16_ER_BL};

/** YUV420sp 8-bit pitch-linear format with limited range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 16 to 240. Resolution is half of luma plane,
     *     both horizontally and vertically.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV12{NVCV_IMAGE_FORMAT_NV12};

/** YUV420sp 8-bit block-linear format with limited range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 0 to 255. Resolution is half of luma plane,
     *     both horizontally and vertically.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV12_BL{NVCV_IMAGE_FORMAT_NV12_BL};

/** YUV420sp 8-bit pitch-linear format with full range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 0 to 255. Resolution is half of luma plane,
     *     both horizontally and vertically.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV12_ER{NVCV_IMAGE_FORMAT_NV12_ER};

/** YUV420sp 8-bit block-linear format with full range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 0 to 255. Resolution is half of luma plane,
     *     both horizontally and vertically.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV12_ER_BL{NVCV_IMAGE_FORMAT_NV12_ER_BL};

/** YUV444sp 8-bit pitch-linear format with limited range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 16 to 240. It has the same resolution as luma plane.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV24{NVCV_IMAGE_FORMAT_NV24};

/** YUV444sp 8-bit block-linear format with limited range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 16 to 235.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 0 to 255. It has the same resolution as luma plane.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV24_BL{NVCV_IMAGE_FORMAT_NV24_BL};

/** YUV444sp 8-bit pitch-linear format with full range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 0 to 255. It has the same resolution as luma plane.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV24_ER{NVCV_IMAGE_FORMAT_NV24_ER};

/** YUV444sp 8-bit block-linear format with full range.
     *  Format is composed of two planes:
     *  1. One 8-bit channel with luma (Y'). Values range from 0 to 255.
     *  2. Two interleaved 8-bit channels with chroma (Cb,Cr).
     *     Values range from 0 to 255. It has the same resolution as luma plane.
     *     For a given pixel, Cb channel has lower memory address than Cr.
     */
constexpr ImageFormat FMT_NV24_ER_BL{NVCV_IMAGE_FORMAT_NV24_ER_BL};

/** YUV422 8-bit pitch-linear format in one plane with UYVY ordering and limited range. */
constexpr ImageFormat FMT_UYVY{NVCV_IMAGE_FORMAT_UYVY};

/** YUV422 8-bit block-linear format in one plane with UYVY ordering and limited range. */
constexpr ImageFormat FMT_UYVY_BL{NVCV_IMAGE_FORMAT_UYVY_BL};

/** YUV422 8-bit pitch-linear format in one plane with UYVY ordering and full range. */
constexpr ImageFormat FMT_UYVY_ER{NVCV_IMAGE_FORMAT_UYVY_ER};

/** YUV422 8-bit block-linear format in one plane with UYVY ordering and full range. */
constexpr ImageFormat FMT_UYVY_ER_BL{NVCV_IMAGE_FORMAT_UYVY_ER_BL};

/** YUV422 8-bit pitch-linear format in one plane with YUYV ordering and limited range.
     *  Also known as YUY2 format.
     */
constexpr ImageFormat FMT_YUYV{NVCV_IMAGE_FORMAT_YUYV};

/** YUV422 8-bit block-linear format in one plane with YUYV ordering and limited range.
     *  Also known as YUY2 format.
     */
constexpr ImageFormat FMT_YUYV_BL{NVCV_IMAGE_FORMAT_YUYV_BL};

/** YUV422 8-bit pitch-linear format in one plane with YUYV ordering and full range.
     *  Also known as YUY2 format.
     */
constexpr ImageFormat FMT_YUYV_ER{NVCV_IMAGE_FORMAT_YUYV_ER};

/** YUV422 8-bit block-linear format in one plane with YUYV ordering and full range.
     *  Also known as YUY2 format.
     */
constexpr ImageFormat FMT_YUYV_ER_BL{NVCV_IMAGE_FORMAT_YUYV_ER_BL};

/** Single plane with interleaved RGB 8-bit channel. */
constexpr ImageFormat FMT_RGB8{NVCV_IMAGE_FORMAT_RGB8};

/** Single plane with interleaved BGR 8-bit channel. */
constexpr ImageFormat FMT_BGR8{NVCV_IMAGE_FORMAT_BGR8};

/** Single plane with interleaved RGBA 8-bit channel. */
constexpr ImageFormat FMT_RGBA8{NVCV_IMAGE_FORMAT_RGBA8};

/** Single plane with interleaved BGRA 8-bit channel. */
constexpr ImageFormat FMT_BGRA8{NVCV_IMAGE_FORMAT_BGRA8};

inline ImageFormat::ImageFormat(ColorSpec colorSpec, ChromaSubsampling chromaSub, MemLayout memLayout,
                                DataType dataType, Swizzle swizzle, Packing packing0, Packing packing1,
                                Packing packing2, Packing packing3)
{
    detail::CheckThrow(nvcvMakeYCbCrImageFormat(
        &m_format, static_cast<NVCVColorSpec>(colorSpec), static_cast<NVCVChromaSubsampling>(chromaSub),
        static_cast<NVCVMemLayout>(memLayout), static_cast<NVCVDataType>(dataType), static_cast<NVCVSwizzle>(swizzle),
        static_cast<NVCVPacking>(packing0), static_cast<NVCVPacking>(packing1), static_cast<NVCVPacking>(packing2),
        static_cast<NVCVPacking>(packing3)));
}

constexpr ImageFormat ImageFormat::ConstCreate(ColorSpec colorSpec, ChromaSubsampling chromaSub, MemLayout memLayout,
                                               DataType dataType, Swizzle swizzle, Packing packing0, Packing packing1,
                                               Packing packing2, Packing packing3)
{
    return ImageFormat{NVCV_MAKE_YCbCr_IMAGE_FORMAT(
        static_cast<NVCVColorSpec>(colorSpec), static_cast<NVCVChromaSubsampling>(chromaSub),
        static_cast<NVCVMemLayout>(memLayout), static_cast<NVCVDataType>(dataType), static_cast<NVCVSwizzle>(swizzle),
        4, static_cast<NVCVPacking>(packing0), static_cast<NVCVPacking>(packing1), static_cast<NVCVPacking>(packing2),
        static_cast<NVCVPacking>(packing3))};
}

inline ImageFormat::ImageFormat(ColorModel colorModel, ColorSpec colorSpec, MemLayout memLayout, DataType dataType,
                                Swizzle swizzle, Packing packing0, Packing packing1, Packing packing2, Packing packing3)
{
    detail::CheckThrow(nvcvMakeColorImageFormat(
        &m_format, static_cast<NVCVColorModel>(colorModel), static_cast<NVCVColorSpec>(colorSpec),
        static_cast<NVCVMemLayout>(memLayout), static_cast<NVCVDataType>(dataType), static_cast<NVCVSwizzle>(swizzle),
        static_cast<NVCVPacking>(packing0), static_cast<NVCVPacking>(packing1), static_cast<NVCVPacking>(packing2),
        static_cast<NVCVPacking>(packing3)));
}

constexpr ImageFormat ImageFormat::ConstCreate(ColorModel colorModel, ColorSpec colorSpec, MemLayout memLayout,
                                               DataType dataType, Swizzle swizzle, Packing packing0, Packing packing1,
                                               Packing packing2, Packing packing3)
{
    return ImageFormat{NVCV_MAKE_COLOR_IMAGE_FORMAT(
        static_cast<NVCVColorModel>(colorModel), static_cast<NVCVColorSpec>(colorSpec),
        static_cast<NVCVMemLayout>(memLayout), static_cast<NVCVDataType>(dataType), static_cast<NVCVSwizzle>(swizzle),
        4, static_cast<NVCVPacking>(packing0), static_cast<NVCVPacking>(packing1), static_cast<NVCVPacking>(packing2),
        static_cast<NVCVPacking>(packing3))};
}

inline ImageFormat::ImageFormat(MemLayout memLayout, DataType dataType, Swizzle swizzle, Packing packing0,
                                Packing packing1, Packing packing2, Packing packing3)
{
    detail::CheckThrow(nvcvMakeNonColorImageFormat(
        &m_format, static_cast<NVCVMemLayout>(memLayout), static_cast<NVCVDataType>(dataType),
        static_cast<NVCVSwizzle>(swizzle), static_cast<NVCVPacking>(packing0), static_cast<NVCVPacking>(packing1),
        static_cast<NVCVPacking>(packing2), static_cast<NVCVPacking>(packing3)));
}

constexpr ImageFormat ImageFormat::ConstCreate(MemLayout memLayout, DataType dataType, Swizzle swizzle,
                                               Packing packing0, Packing packing1, Packing packing2, Packing packing3)
{
    return ImageFormat{NVCV_MAKE_NONCOLOR_IMAGE_FORMAT(
        static_cast<NVCVMemLayout>(memLayout), static_cast<NVCVDataType>(dataType), static_cast<NVCVSwizzle>(swizzle),
        4, static_cast<NVCVPacking>(packing0), static_cast<NVCVPacking>(packing1), static_cast<NVCVPacking>(packing2),
        static_cast<NVCVPacking>(packing3))};
}

inline ImageFormat::ImageFormat(RawPattern rawPattern, MemLayout memLayout, DataType dataType, Swizzle swizzle,
                                Packing packing0, Packing packing1, Packing packing2, Packing packing3)
{
    detail::CheckThrow(nvcvMakeRawImageFormat(
        &m_format, static_cast<NVCVRawPattern>(rawPattern), static_cast<NVCVMemLayout>(memLayout),
        static_cast<NVCVDataType>(dataType), static_cast<NVCVSwizzle>(swizzle), static_cast<NVCVPacking>(packing0),
        static_cast<NVCVPacking>(packing1), static_cast<NVCVPacking>(packing2), static_cast<NVCVPacking>(packing3)));
}

constexpr ImageFormat ImageFormat::ConstCreate(RawPattern rawPattern, MemLayout memLayout, DataType dataType,
                                               Swizzle swizzle, Packing packing0, Packing packing1, Packing packing2,
                                               Packing packing3)
{
    return ImageFormat{NVCV_MAKE_RAW_IMAGE_FORMAT(
        static_cast<NVCVRawPattern>(rawPattern), static_cast<NVCVMemLayout>(memLayout),
        static_cast<NVCVDataType>(dataType), static_cast<NVCVSwizzle>(swizzle), 4, static_cast<NVCVPacking>(packing0),
        static_cast<NVCVPacking>(packing1), static_cast<NVCVPacking>(packing2), static_cast<NVCVPacking>(packing3))};
}

inline ImageFormat ImageFormat::FromPlanes(ImageFormat plane0, ImageFormat plane1, ImageFormat plane2,
                                           ImageFormat plane3)
{
    NVCVImageFormat fmt;

    detail::CheckThrow(nvcvMakeImageFormatFromPlanes(&fmt, plane0, plane1, plane2, plane3));
    return ImageFormat{fmt};
}

inline RawPattern ImageFormat::rawPattern() const noexcept
{
    NVCVRawPattern out;
    detail::CheckThrow(nvcvImageFormatGetRawPattern(m_format, &out));
    return static_cast<RawPattern>(out);
}

inline bool HasSameDataLayout(ImageFormat a, ImageFormat b)
{
    int8_t out;
    detail::CheckThrow(nvcvImageFormatHasSameDataLayout(a, b, &out));
    return out != 0;
}

inline ImageFormat ImageFormat::FromFourCC(uint fourcc, ColorSpec colorSpec, MemLayout memLayout)
{
    NVCVImageFormat out;
    detail::CheckThrow(nvcvMakeImageFormatFromFourCC(&out, fourcc, static_cast<NVCVColorSpec>(colorSpec),
                                                     static_cast<NVCVMemLayout>(memLayout)));
    return ImageFormat{out};
}

inline uint32_t ImageFormat::fourCC() const
{
    uint32_t out;
    detail::CheckThrow(nvcvImageFormatToFourCC(m_format, &out));
    return out;
}

constexpr ImageFormat::operator NVCVImageFormat() const noexcept
{
    return m_format;
}

constexpr NVCVImageFormat ImageFormat::cvalue() const noexcept
{
    return m_format;
}

constexpr bool ImageFormat::operator==(ImageFormat that) const noexcept
{
    return m_format == that.cvalue();
}

constexpr bool ImageFormat::operator!=(ImageFormat that) const noexcept
{
    return !operator==(that);
}

inline Packing ImageFormat::planePacking(int plane) const noexcept
{
    NVCVPacking out;
    detail::CheckThrow(nvcvImageFormatGetPlanePacking(m_format, plane, &out));
    return static_cast<Packing>(out);
}

inline Size2D ImageFormat::planeSize(Size2D imgSize, int plane) const noexcept
{
    Size2D psize;
    detail::CheckThrow(nvcvImageFormatGetPlaneSize(m_format, plane, imgSize.w, imgSize.h, &psize.w, &psize.h));
    return psize;
}

inline int32_t ImageFormat::planeBitsPerPixel(int plane) const noexcept
{
    int32_t out;
    detail::CheckThrow(nvcvImageFormatGetPlaneBitsPerPixel(m_format, plane, &out));
    return out;
}

inline DataType ImageFormat::dataType() const noexcept
{
    NVCVDataType out;
    detail::CheckThrow(nvcvImageFormatGetDataType(m_format, &out));
    return static_cast<DataType>(out);
}

inline Swizzle ImageFormat::swizzle() const noexcept
{
    NVCVSwizzle out;
    detail::CheckThrow(nvcvImageFormatGetSwizzle(m_format, &out));
    return static_cast<Swizzle>(out);
}

inline Swizzle ImageFormat::planeSwizzle(int plane) const noexcept
{
    NVCVSwizzle out;
    detail::CheckThrow(nvcvImageFormatGetPlaneSwizzle(m_format, plane, &out));
    return static_cast<Swizzle>(out);
}

inline MemLayout ImageFormat::memLayout() const noexcept
{
    NVCVMemLayout out;
    detail::CheckThrow(nvcvImageFormatGetMemLayout(m_format, &out));
    return static_cast<MemLayout>(out);
}

inline ColorSpec ImageFormat::colorSpec() const noexcept
{
    NVCVColorSpec out;
    detail::CheckThrow(nvcvImageFormatGetColorSpec(m_format, &out));
    return static_cast<ColorSpec>(out);
}

inline ColorModel ImageFormat::colorModel() const noexcept
{
    NVCVColorModel out;
    detail::CheckThrow(nvcvImageFormatGetColorModel(m_format, &out));
    return static_cast<ColorModel>(out);
}

inline ChromaSubsampling ImageFormat::chromaSubsampling() const noexcept
{
    NVCVChromaSubsampling out;
    detail::CheckThrow(nvcvImageFormatGetChromaSubsampling(m_format, &out));
    return static_cast<ChromaSubsampling>(out);
}

inline int32_t ImageFormat::planeNumChannels(int plane) const noexcept
{
    int32_t out;
    detail::CheckThrow(nvcvImageFormatGetPlaneNumChannels(m_format, plane, &out));
    return out;
}

inline int32_t ImageFormat::numPlanes() const noexcept
{
    int32_t out;
    detail::CheckThrow(nvcvImageFormatGetNumPlanes(m_format, &out));
    return out;
}

inline int32_t ImageFormat::numChannels() const noexcept
{
    int32_t out;
    detail::CheckThrow(nvcvImageFormatGetNumChannels(m_format, &out));
    return out;
}

inline std::array<int32_t, 4> ImageFormat::bitsPerChannel() const noexcept
{
    std::array<int32_t, 4> out;
    detail::CheckThrow(nvcvImageFormatGetBitsPerChannel(m_format, &out[0]));
    return out;
}

inline PixelType ImageFormat::planePixelType(int plane) const noexcept
{
    NVCVPixelType out;
    detail::CheckThrow(nvcvImageFormatGetPlanePixelType(m_format, plane, &out));
    return static_cast<PixelType>(out);
}

inline ImageFormat ImageFormat::swizzleAndPacking(Swizzle newSwizzle, Packing newPacking0, Packing newPacking1,
                                                  Packing newPacking2, Packing newPacking3) const
{
    NVCVImageFormat out = m_format;
    detail::CheckThrow(nvcvImageFormatSetSwizzleAndPacking(
        &out, static_cast<NVCVSwizzle>(newSwizzle), static_cast<NVCVPacking>(newPacking0),
        static_cast<NVCVPacking>(newPacking1), static_cast<NVCVPacking>(newPacking2),
        static_cast<NVCVPacking>(newPacking3)));
    return ImageFormat{out};
}

inline ImageFormat ImageFormat::dataType(DataType newDataType) const
{
    NVCVImageFormat out = m_format;
    detail::CheckThrow(nvcvImageFormatSetDataType(&out, static_cast<NVCVDataType>(newDataType)));
    return ImageFormat{out};
}

inline ImageFormat ImageFormat::memLayout(MemLayout newMemLayout) const
{
    NVCVImageFormat out = m_format;
    detail::CheckThrow(nvcvImageFormatSetMemLayout(&out, static_cast<NVCVMemLayout>(newMemLayout)));
    return ImageFormat{out};
}

inline ImageFormat ImageFormat::colorSpec(ColorSpec newColorSpec) const
{
    NVCVImageFormat out = m_format;
    detail::CheckThrow(nvcvImageFormatSetColorSpec(&out, static_cast<NVCVColorSpec>(newColorSpec)));
    return ImageFormat{out};
}

inline ImageFormat ImageFormat::chromaSubsampling(ChromaSubsampling newCSS) const
{
    NVCVImageFormat out = m_format;
    detail::CheckThrow(nvcvImageFormatSetChromaSubsampling(&out, static_cast<NVCVChromaSubsampling>(newCSS)));
    return ImageFormat{out};
}

inline ImageFormat ImageFormat::rawPattern(RawPattern newRawPattern) const
{
    NVCVImageFormat out = m_format;
    detail::CheckThrow(nvcvImageFormatSetRawPattern(&out, static_cast<NVCVRawPattern>(newRawPattern)));
    return ImageFormat{out};
}

inline std::ostream &operator<<(std::ostream &out, ImageFormat fmt)
{
    return out << nvcvImageFormatGetName(fmt);
}

} // namespace nv::cv

#endif // NVCV_IMAGE_FORMAT_HPP
