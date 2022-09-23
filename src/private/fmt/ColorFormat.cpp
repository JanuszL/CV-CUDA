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

#include "ColorFormat.hpp"

namespace nv::cv::priv {

bool operator==(const ColorFormat &a, const ColorFormat &b)
{
    if (a.model == b.model)
    {
        if (a.model == NVCV_COLOR_MODEL_RAW)
        {
            return a.raw == b.raw;
        }
        else
        {
            return a.cspec == b.cspec;
        }
    }
    else
    {
        return false;
    }
}

bool operator!=(const ColorFormat &a, const ColorFormat &b)
{
    return !operator==(a, b);
}

} // namespace nv::cv::priv
