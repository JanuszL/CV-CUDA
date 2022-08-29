/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <nvcv/Foo.hpp>

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace nv::cv {

namespace {

__global__ void gpuFoo(int value, int *out)
{
    *out = value;
}

void CheckCudaError(cudaError_t err, std::string msg)
{
    if (err != cudaSuccess)
    {
        std::ostringstream ss;
        ss << msg << ": " << cudaGetErrorName(err) << " - " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

} // namespace

bool Foo(int value)
{
    if (value == 0)
    {
        throw std::invalid_argument("Invalid value parameter, it cannot be 0");
    }

    int result;

    int *devPtr;
    CheckCudaError(cudaMalloc(&devPtr, sizeof(*devPtr)), "Error allocating CUDA memory");

    try
    {
        gpuFoo<<<dim3(1, 1), dim3(1, 1)>>>(value, devPtr);

        CheckCudaError(cudaDeviceSynchronize(), "Error launchign kernel");

        CheckCudaError(cudaMemcpy(&result, devPtr, sizeof(int), cudaMemcpyDeviceToHost),
                       "Error copying memory from device");

        if (result != value)
        {
            throw std::runtime_error("Memory copied to host doesn't match contents in device memory");
        }
    }
    catch (...)
    {
        cudaFree(devPtr);
        throw;
    }

    cudaFree(devPtr);

    return result == 42;
}

} // namespace nv::cv
