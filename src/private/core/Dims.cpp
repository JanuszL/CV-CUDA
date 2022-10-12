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

#include "Dims.hpp"

#include <util/Assert.h>

namespace nv::cv::priv {

DimsNCHW ToNCHW(const Shape &s, NVCVTensorLayout layout)
{
    switch (layout)
    {
    case NVCV_TENSOR_NCHW:
        return {s[0], s[1], s[2], s[3]};

    case NVCV_TENSOR_NHWC:
        return {s[0], s[3], s[1], s[2]};
    }

    NVCV_ASSERT(!"Invalid layout");
    return {};
}

} // namespace nv::cv::priv
