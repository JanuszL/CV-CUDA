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

#include "TensorShape.hpp"

#include "Exception.hpp"
#include "TensorLayout.hpp"

namespace nv::cv::priv {

void PermuteShape(const NVCVTensorLayout &srcLayout, const int64_t *srcShape, const NVCVTensorLayout &dstLayout,
                  int64_t *dstShape)
{
    if (srcShape == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to source shape cannot be NULL");
    }

    if (dstShape == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to destination shape cannot be NULL");
    }

    std::fill_n(dstShape, dstLayout.ndim, 1);

    for (int i = 0; i < srcLayout.ndim; ++i)
    {
        int dstIdx = FindDimIndex(dstLayout, srcLayout.data[i]);
        if (dstIdx >= 0)
        {
            dstShape[dstIdx] = srcShape[i];
        }
    }
}

} // namespace nv::cv::priv
