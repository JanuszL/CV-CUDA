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

#ifndef NVCV_PRIV_TENSORLAYOUT_HPP
#define NVCV_PRIV_TENSORLAYOUT_HPP

#include <nvcv/TensorData.h>

namespace nv::cv::priv {

NVCVTensorLayout CreateLayout(const char *descr);
NVCVTensorLayout CreateLayout(const char *beg, const char *end);

NVCVTensorLayout CreateFirst(const NVCVTensorLayout &layout, int n);
NVCVTensorLayout CreateLast(const NVCVTensorLayout &layout, int n);
NVCVTensorLayout CreateSubRange(const NVCVTensorLayout &layout, int beg, int end);

int FindDimIndex(const NVCVTensorLayout &layout, char dimLabel);

bool IsChannelLast(const NVCVTensorLayout &layout);

bool operator==(const NVCVTensorLayout &a, const NVCVTensorLayout &b);
bool operator!=(const NVCVTensorLayout &a, const NVCVTensorLayout &b);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TENSORLAYOUT_HPP
