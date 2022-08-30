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

#ifndef NVCV_TEST_COMMON_PRINTERS_HPP
#define NVCV_TEST_COMMON_PRINTERS_HPP

#include <core/Status.hpp>
#include <cuda_runtime.h>
#include <fmt/Printers.hpp>
#include <nvcv/Status.h>

#include <iosfwd>

std::ostream &operator<<(std::ostream &out, NVCVStatus status);
std::ostream &operator<<(std::ostream &out, cudaError_t err);

#endif // NVCV_TEST_COMMON_PRINTERS_HPP
