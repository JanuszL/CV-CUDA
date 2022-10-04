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

#include "AllocInfo.hpp"

#include <cuda_runtime.h>
#include <fmt/ImageFormat.hpp>
#include <fmt/PixelType.hpp>
#include <util/Assert.h>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <numeric>

namespace nv::cv::priv {

// Images -----------------------------------------

bool AllocInfo2D::PlaneInfo::operator==(const AllocInfo2D::PlaneInfo &that) const
{
    return this->offsetBytes == that.offsetBytes && this->rowPitchBytes == that.rowPitchBytes;
}

bool AllocInfo2D::operator==(const AllocInfo2D &that) const
{
    if (this->size == that.size && this->alignment == that.alignment)
    {
        if (this->planes.size() == that.planes.size())
        {
            if (std::equal(this->planes.begin(), this->planes.end(), that.planes.begin()))
            {
                return true;
            }
        }
    }
    return false;
}

bool AllocInfo2D::operator!=(const AllocInfo2D &that) const
{
    return !(*this == that);
}

AllocInfo2D CalcAllocInfo(Size2D size, ImageFormat fmt)
{
    AllocInfo2D out;

    int dev;
    NVCV_CHECK_THROW(cudaGetDevice(&dev));

    int pitchAlign;
    NVCV_CHECK_THROW(cudaDeviceGetAttribute(&pitchAlign, cudaDevAttrTexturePitchAlignment, dev));
    NVCV_CHECK_THROW(cudaDeviceGetAttribute(&out.alignment, cudaDevAttrTextureAlignment, dev));

    // Alignment must be compatible with each plane's pixel stride.
    for (int p = 0; p < fmt.numPlanes(); ++p)
    {
        int stride = fmt.planePixelStrideBytes(p);

        out.alignment = std::lcm(out.alignment, stride);
        pitchAlign    = std::lcm(pitchAlign, stride);
    }

    out.size = 0;
    for (int p = 0; p < fmt.numPlanes(); ++p)
    {
        Size2D planeSize = fmt.planeSize(size, p);

        AllocInfo2D::PlaneInfo pinfo;

        pinfo.rowPitchBytes = util::RoundUp((size_t)planeSize.w * fmt.planePixelStrideBytes(p), pitchAlign);
        pinfo.offsetBytes   = out.size;

        out.size += util::RoundUp((size_t)pinfo.rowPitchBytes * (size_t)planeSize.h, out.alignment);

        out.planes.push_back(pinfo);
    }

    return out;
}

} // namespace nv::cv::priv
