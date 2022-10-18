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

#include "DeviceSaturateCast.hpp" // to test in the device

#include <gtest/gtest.h>              // for EXPECT_EQ, etc.
#include <nvcv/cuda/SaturateCast.hpp> // the object of this test

namespace cuda = nv::cv::cuda;

// ----------------- To allow testing device-side SaturateCast -----------------

template<typename TargetPixelType, typename SourcePixelType>
__global__ void RunSaturateCast(TargetPixelType *out, SourcePixelType u)
{
    out[0] = cuda::SaturateCast<cuda::BaseType<TargetPixelType>>(u);
}

template<typename TargetPixelType, typename SourcePixelType>
TargetPixelType DeviceRunSaturateCast(SourcePixelType pix)
{
    TargetPixelType *dTest;
    TargetPixelType  hTest[1];

    EXPECT_EQ(cudaSuccess, cudaMalloc(&dTest, sizeof(TargetPixelType)));

    RunSaturateCast<<<1, 1>>>(dTest, pix);

    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    EXPECT_EQ(cudaSuccess, cudaMemcpy(hTest, dTest, sizeof(TargetPixelType), cudaMemcpyDeviceToHost));

    EXPECT_EQ(cudaSuccess, cudaFree(dTest));

    return hTest[0];
}

// Need to instantiate each test on TestSaturateCast, making sure not to use const types

#define NVCV_TEST_INST(TARGET_PIXEL_TYPE, SOURCE_PIXEL_TYPE) \
    template TARGET_PIXEL_TYPE DeviceRunSaturateCast(SOURCE_PIXEL_TYPE pix)

NVCV_TEST_INST(char, char);
NVCV_TEST_INST(short, short);
NVCV_TEST_INST(int, int);
NVCV_TEST_INST(float, float);
NVCV_TEST_INST(double, double);

NVCV_TEST_INST(float3, double3);
NVCV_TEST_INST(double3, float3);

NVCV_TEST_INST(float4, char4);
NVCV_TEST_INST(float3, ushort3);
NVCV_TEST_INST(double2, uchar2);
NVCV_TEST_INST(double2, int2);

NVCV_TEST_INST(char2, float2);
NVCV_TEST_INST(ushort2, float2);
NVCV_TEST_INST(int2, float2);
NVCV_TEST_INST(uint2, float2);
NVCV_TEST_INST(uchar2, double2);
NVCV_TEST_INST(char2, double2);
NVCV_TEST_INST(short2, double2);

NVCV_TEST_INST(short1, char1);
NVCV_TEST_INST(ulonglong2, ulong2);
NVCV_TEST_INST(longlong2, long2);
NVCV_TEST_INST(ushort3, char3);
NVCV_TEST_INST(short2, uchar2);
NVCV_TEST_INST(uchar4, char4);
NVCV_TEST_INST(char3, uchar3);

NVCV_TEST_INST(short1, int1);
NVCV_TEST_INST(short2, uint2);
NVCV_TEST_INST(ushort3, int3);
NVCV_TEST_INST(uchar2, int2);
NVCV_TEST_INST(char2, uint2);
NVCV_TEST_INST(uchar2, ulonglong2);
NVCV_TEST_INST(char2, long2);

#undef NVCV_TEST_INST
