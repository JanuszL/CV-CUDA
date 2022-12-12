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

#ifndef NVCV_TEST_COMMON_FLIP_UTILS_HPP
#define NVCV_TEST_COMMON_FLIP_UTILS_HPP

#include <cuda_runtime.h>       // for long3, etc.
#include <nvcv/ImageFormat.hpp> // for ImageFormat, etc.
#include <nvcv/Size.hpp>        // for Size2D, etc.

#include <cstdint> // for uint8_t, etc.
#include <vector>  // for std::vector, etc.

namespace nv::cv::test {

void FlipCPU(std::vector<uint8_t> &hDst, const long3 &dstPitches, const std::vector<uint8_t> &hSrc,
             const long3 &srcPitches, const int3 &shape, const ImageFormat &format, int flipCode);

} // namespace nv::cv::test

#endif // NVCV_TEST_COMMON_FLIP_UTILS_HPP
