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

#include "CheckError.hpp"

#include <iostream>
#include <sstream>

namespace nv::cvpy {

static std::string ToString(cudaError_t err)
{
    std::ostringstream ss;
    ss << cudaGetErrorName(err) << ": " << cudaGetErrorString(err);
    return ss.str();
}

void CheckThrow(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // consume the error
        throw std::runtime_error(ToString(err));
    }
}

void CheckLog(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cudaGetLastError(); // consume the error
        std::cerr << ToString(err) << std::endl;
    }
}

} // namespace nv::cvpy
