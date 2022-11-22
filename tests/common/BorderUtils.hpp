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

#ifndef NVCV_TEST_COMMON_BORDER_UTILS_HPP
#define NVCV_TEST_COMMON_BORDER_UTILS_HPP

#include <cuda_runtime.h>    // for int2, etc.
#include <operators/Types.h> // for NVCVBorderType, etc.

namespace nv::cv::test {

void ReplicateBorderIndex(int2 &coord, int2 size);

void WrapBorderIndex(int2 &coord, int2 size);

void ReflectBorderIndex(int2 &coord, int2 size);

void Reflect101BorderIndex(int2 &coord, int2 size);

bool IsInside(int2 &inCoord, int2 inSize, NVCVBorderType borderMode);

} // namespace nv::cv::test

#endif // NVCV_TEST_COMMON_BORDER_UTILS_HPP
