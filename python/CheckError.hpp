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

#ifndef NVCV_PYTHON_CHECKERROR_HPP
#define NVCV_PYTHON_CHECKERROR_HPP

#include <cuda_runtime.h>
#include <nvcv/detail/CheckError.hpp>

namespace nv::cvpy {

using nv::cv::detail::CheckThrow;

void CheckThrow(cudaError_t err);

void CheckLog(cudaError_t err);

} // namespace nv::cvpy

#endif // NVCV_PYTHON_CHECKERROR_HPP
