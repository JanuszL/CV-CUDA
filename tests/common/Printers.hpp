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

#include <cuda_runtime.h>
#include <fmt/Printers.hpp>

#include <iostream>

#if NVCV_EXPORTING
#    include <core/Status.hpp>
#else
#    include <nvcv/Status.hpp>
#endif

#if NVCV_EXPORTING
inline std::ostream &operator<<(std::ostream &out, NVCVStatus status)
{
    return out << nv::cv::priv::GetName(status);
}
#endif

std::ostream &operator<<(std::ostream &out, cudaError_t err);

#endif // NVCV_TEST_COMMON_PRINTERS_HPP
