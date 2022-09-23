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

#ifndef NVCV_PRIV_PIXEL_TYPE_HPP
#define NVCV_PRIV_PIXEL_TYPE_HPP

#include "ImageFormat.hpp"

#include <nvcv/PixelType.h>

#include <string>

namespace nv::cv::priv {

// Wrapper to NVCVPixelType to make it properly typed.
class PixelType
{
public:
    constexpr explicit PixelType(NVCVPixelType type)
        : m_type{type}
    {
    }

    constexpr PixelType(NVCVMemLayout memLayout, NVCVDataType dataType, NVCVPacking packing) noexcept
        : m_type{NVCV_MAKE_PIXEL_TYPE(memLayout, dataType, packing)}
    {
    }

    constexpr NVCVPixelType value() const
    {
        return m_type;
    }

    constexpr bool operator==(PixelType that) const
    {
        return m_type == that.value();
    }

    constexpr bool operator!=(PixelType that) const
    {
        return !operator==(that);
    }

    constexpr NVCVPacking packing() const noexcept
    {
        return ImageFormat{m_type}.planePacking(0);
    }

    constexpr NVCVDataType dataType() const noexcept
    {
        return ImageFormat{m_type}.dataType();
    }

    PixelType     memLayout(NVCVMemLayout newMemLayout) const;
    NVCVMemLayout memLayout() const noexcept;

    int                    bpp() const noexcept;
    std::array<int32_t, 4> bpc() const noexcept;
    int                    numChannels() const noexcept;
    int                    elemStride() const noexcept;
    PixelType              channelType(int ch) const;

private:
    NVCVPixelType m_type;
};

std::ostream &operator<<(std::ostream &out, PixelType type);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_PIXEL_TYPE_HPP
