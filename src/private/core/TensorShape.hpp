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

#ifndef NVCV_PRIV_TENSORSHAPE_HPP
#define NVCV_PRIV_TENSORSHAPE_HPP

#include <nvcv/TensorLayout.h>
#include <util/StaticVector.hpp>

namespace nv::cv::priv {

void PermuteShape(const NVCVTensorLayout &srcLayout, const int64_t *srcShape, const NVCVTensorLayout &dstLayout,
                  int64_t *dstShape);

}

#endif // NVCV_PRIV_TENSORSHAPE_HPP
