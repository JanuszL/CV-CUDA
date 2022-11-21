/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

    constexpr PixelType(NVCVDataKind dataKind, NVCVPacking packing) noexcept
        : m_type{NVCV_MAKE_PIXEL_TYPE(dataKind, packing)}
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

    constexpr NVCVDataKind dataKind() const noexcept
    {
        return ImageFormat{m_type}.dataKind();
    }

    int                    bpp() const noexcept;
    std::array<int32_t, 4> bpc() const noexcept;
    int                    numChannels() const noexcept;
    int                    strideBytes() const noexcept;
    int                    alignment() const noexcept;
    PixelType              channelType(int ch) const;

private:
    NVCVPixelType m_type;
};

std::ostream &operator<<(std::ostream &out, PixelType type);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_PIXEL_TYPE_HPP
