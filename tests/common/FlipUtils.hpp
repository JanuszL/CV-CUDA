/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_TEST_COMMON_FLIP_UTILS_HPP
#define NVCV_TEST_COMMON_FLIP_UTILS_HPP

#include <cuda_runtime.h>       // for long3, etc.
#include <nvcv/ImageFormat.hpp> // for ImageFormat, etc.
#include <nvcv/Size.hpp>        // for Size2D, etc.

#include <cstdint> // for uint8_t, etc.
#include <vector>  // for std::vector, etc.

namespace nv::cv::test {

void FlipCPU(std::vector<uint8_t> &hDst, const long3 &dstStrides, const std::vector<uint8_t> &hSrc,
             const long3 &srcStrides, const int3 &shape, const ImageFormat &format, int flipCode);

} // namespace nv::cv::test

#endif // NVCV_TEST_COMMON_FLIP_UTILS_HPP
