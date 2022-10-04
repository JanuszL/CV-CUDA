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

#ifndef NVCV_PRIV_ALLOCINFO_HPP
#define NVCV_PRIV_ALLOCINFO_HPP

#include "Size.hpp"

#include <fmt/ImageFormat.hpp>
#include <nvcv/Image.h>
#include <util/StaticVector.hpp>

#include <utility>

namespace nv::cv::priv {

struct AllocInfo2D
{
    size_t size      = 0;
    int    alignment = 0;

    struct PlaneInfo
    {
        int offsetBytes   = 0;
        int rowPitchBytes = 0;

        bool operator==(const PlaneInfo &that) const;
    };

    util::StaticVector<PlaneInfo, NVCV_MAX_PLANE_COUNT> planes;

    bool operator==(const AllocInfo2D &that) const;
    bool operator!=(const AllocInfo2D &that) const;
};

AllocInfo2D CalcAllocInfo(Size2D size, ImageFormat fmt);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_ALLOCINFO_HPP
