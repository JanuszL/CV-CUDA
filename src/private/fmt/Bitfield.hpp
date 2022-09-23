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

#ifndef NVCV_PRIV_BITFIELD_HPP
#define NVCV_PRIV_BITFIELD_HPP

#include <cstdint>

namespace nv::cv::priv {

constexpr uint64_t SetBitfield(uint64_t value, int offset, int length) noexcept
{
    return (value & ((1ULL << length) - 1)) << offset;
}

constexpr uint64_t MaskBitfield(int offset, int length) noexcept
{
    return SetBitfield(UINT64_MAX, offset, length);
}

constexpr uint64_t ExtractBitfield(uint64_t value, int offset, int length) noexcept
{
    return (value >> offset) & ((1ULL << length) - 1);
}

} // namespace nv::cv::priv

#endif // NVCV_PRIV_BITFIELD_HPP
