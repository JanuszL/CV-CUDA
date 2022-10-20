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

#include "TensorLayout.hpp"
#include "TensorShape.hpp"

namespace nv::cv::priv {

DimsNCHW ToNCHW(const int64_t *s, const NVCVTensorLayout &layout)
{
    int64_t nchw[4];
    PermuteShape(layout, s, NVCV_TENSOR_NCHW, nchw);

    DimsNCHW dims;
    dims.n = nchw[0];
    dims.c = nchw[1];
    dims.h = nchw[2];
    dims.w = nchw[3];

    return dims;
}

} // namespace nv::cv::priv
