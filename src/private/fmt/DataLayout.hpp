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

#ifndef NVCV_PRIV_DATA_LAYOUT_HPP
#define NVCV_PRIV_DATA_LAYOUT_HPP

#include <nvcv/DataLayout.h>

#include <array>
#include <cstdint>
#include <optional>

namespace nv::cv::priv {

std::optional<NVCVPacking> MakeNVCVPacking(const NVCVPackingParams &params) noexcept;
std::optional<NVCVPacking> MakeNVCVPacking(int bitsX, int bitsY = 0, int bitsZ = 0, int bitsW = 0) noexcept;
NVCVSwizzle MakeNVCVSwizzle(NVCVChannel x, NVCVChannel y = NVCV_CHANNEL_0, NVCVChannel z = NVCV_CHANNEL_0,
                            NVCVChannel w = NVCV_CHANNEL_0) noexcept;

bool IsSubWord(const NVCVPackingParams &p);

int GetBitsPerPixel(NVCVPacking packing) noexcept;

NVCVPackingParams GetPackingParams(NVCVPacking packing) noexcept;

NVCVChannel GetSwizzleChannel(NVCVSwizzle swizzle, int idx) noexcept;

std::array<NVCVChannel, 4> GetChannels(NVCVSwizzle swizzle) noexcept;

int GetNumChannels(NVCVSwizzle swizzle) noexcept;

int GetBlockHeightLog2(NVCVMemLayout memLayout) noexcept;

int GetNumComponents(NVCVPacking packing) noexcept;
int GetNumChannels(NVCVPacking packing) noexcept;

std::array<int32_t, 4> GetBitsPerComponent(NVCVPacking packing) noexcept;

NVCVSwizzle MergePlaneSwizzles(NVCVSwizzle sw0, NVCVSwizzle sw1 = NVCV_SWIZZLE_0000,
                               NVCVSwizzle sw2 = NVCV_SWIZZLE_0000, NVCVSwizzle sw3 = NVCV_SWIZZLE_0000);

// Flips endianness in memory space and return the resulting swizzle.
// Optionally an offset+length in component space can be specified, it'll
// restrict the flipping only to these components alone.
NVCVSwizzle FlipEndianness(NVCVSwizzle swizzle, int off = 0, int len = 4) noexcept;

const char *ToString(NVCVPacking packing);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_DATA_LAYOUT_HPP
