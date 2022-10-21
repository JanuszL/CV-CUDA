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

#include "PixelType.hpp"

#include "DataLayout.hpp"
#include "ImageFormat.hpp"
#include "Printers.hpp"

#include <core/Exception.hpp>
#include <util/Assert.h>

#include <sstream>

namespace nv::cv::priv {

int PixelType::bpp() const noexcept
{
    return ImageFormat{m_type}.planeBPP(0);
}

int PixelType::numChannels() const noexcept
{
    return ImageFormat{m_type}.planeNumChannels(0);
}

int PixelType::strideBytes() const noexcept
{
    return (this->bpp() + 7) / 8;
}

std::array<int32_t, 4> PixelType::bpc() const noexcept
{
    NVCVPacking packing = this->packing();
    return GetBitsPerComponent(packing);
}

PixelType PixelType::channelType(int ch) const
{
    if (ch < 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Channel must be >= 0");
    }

    if (ch >= 4)
    {
        return PixelType{NVCV_PIXEL_TYPE_NONE};
    }

    std::array<int32_t, 4> bits = this->bpc();
    if (bits[ch] == 0)
    {
        return PixelType{NVCV_PIXEL_TYPE_NONE};
    }

    if (std::optional<NVCVPacking> packing = MakeNVCVPacking(bits[ch]))
    {
        return PixelType{this->dataType(), *packing};
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Channel type cannot be represented");
    }
}

std::ostream &operator<<(std::ostream &out, PixelType type)
{
    switch (type.value())
    {
#define NVCV_ENUM(E) \
    case E:          \
        return out << #E;
        NVCV_ENUM(NVCV_PIXEL_TYPE_NONE);

        NVCV_ENUM(NVCV_PIXEL_TYPE_U8);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2U8);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3U8);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4U8);

        NVCV_ENUM(NVCV_PIXEL_TYPE_S8);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2S8);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3S8);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4S8);

        NVCV_ENUM(NVCV_PIXEL_TYPE_U16);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2U16);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3U16);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4U16);

        NVCV_ENUM(NVCV_PIXEL_TYPE_S16);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2S16);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3S16);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4S16);

        NVCV_ENUM(NVCV_PIXEL_TYPE_U32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2U32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3U32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4U32);

        NVCV_ENUM(NVCV_PIXEL_TYPE_S32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2S32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3S32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4S32);

        NVCV_ENUM(NVCV_PIXEL_TYPE_F32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2F32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3F32);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4F32);

        NVCV_ENUM(NVCV_PIXEL_TYPE_U64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2U64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3U64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4U64);

        NVCV_ENUM(NVCV_PIXEL_TYPE_S64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2S64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3S64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4S64);

        NVCV_ENUM(NVCV_PIXEL_TYPE_F64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_2F64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_3F64);
        NVCV_ENUM(NVCV_PIXEL_TYPE_4F64);
#undef NVCV_ENUM
    }

    return out << "NVCVPixelType(" << type.dataType() << "," << type.packing() << ")";
}

} // namespace nv::cv::priv
