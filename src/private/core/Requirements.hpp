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

#ifndef NVCV_PRIV_REQUIREMENTS_HPP
#define NVCV_PRIV_REQUIREMENTS_HPP

#include <nvcv/alloc/Requirements.h>

namespace nv::cv::priv {

void Init(NVCVRequirements &reqs);
void Add(NVCVRequirements &reqSum, const NVCVRequirements &req);

void AddBuffer(NVCVMemRequirements &memReq, int64_t bufSize, int64_t bufAlignment);

int64_t CalcTotalSizeBytes(const NVCVMemRequirements &memReq);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_REQUIREMENTS_HPP
