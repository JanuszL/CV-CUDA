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

#include "DeviceTensorWrap.hpp" // to test in the device

#include <gtest/gtest.h>            // for EXPECT_EQ, etc.
#include <nvcv/cuda/DropCast.hpp>   // for DropCast, etc.
#include <nvcv/cuda/MathOps.hpp>    // for operator == to allow EXPECT_EQ
#include <nvcv/cuda/StaticCast.hpp> // for StaticCast, etc.
#include <nvcv/cuda/TensorWrap.hpp> // the object of this test

namespace cuda = nv::cv::cuda;

// ---------------- To allow testing device-side Tensor2DWrap ------------------

template<typename PixelType>
__global__ void Copy(cuda::Tensor2DWrap<PixelType> dst, cuda::Tensor2DWrap<const PixelType> src)
{
    int2 coord = cuda::StaticCast<int>(cuda::DropCast<2>(threadIdx));
    dst[coord] = src[coord];
}

template<typename PixelType, int H, int W>
void DeviceUseTensor2DWrap(PackedImage<PixelType, H, W> &hGold)
{
    PixelType *dInput;
    PixelType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, H * W * sizeof(PixelType)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, H * W * sizeof(PixelType)));

    cuda::Tensor2DWrap<const PixelType> src(dInput, hGold.pitchBytes);
    cuda::Tensor2DWrap<PixelType>       dst(dTest, hGold.pitchBytes);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), H * W * sizeof(PixelType), cudaMemcpyHostToDevice));

    Copy<<<1, dim3(W, H)>>>(dst, src);

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    PackedImage<PixelType, H, W> hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, H * W * sizeof(PixelType), cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestMem2DPackedWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(PIXEL_TYPE, H, W) template void DeviceUseTensor2DWrap(PackedImage<PIXEL_TYPE, H, W> &)

NVCV_TEST_INST_USE(int, 2, 2);
NVCV_TEST_INST_USE(short3, 1, 2);
NVCV_TEST_INST_USE(float1, 2, 4);
NVCV_TEST_INST_USE(uchar4, 3, 3);

#undef NVCV_TEST_INST_USE

template<typename PixelType>
__global__ void SetOnes(nv::cv::cuda::Tensor2DWrap<PixelType> dst, int2 size)
{
    int2 coord = cuda::StaticCast<int>(cuda::DropCast<2>(blockIdx * blockDim + threadIdx));

    if (coord.x >= size.x || coord.y >= size.y)
    {
        return;
    }

    dst[coord] = cuda::SetAll<PixelType>(1);
}

template<typename PixelType>
void DeviceSetOnes(nv::cv::cuda::Tensor2DWrap<PixelType> &wrap, int2 size, cudaStream_t &stream)
{
    dim3 block{32, 4};
    dim3 grid{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y};

    SetOnes<<<grid, block, 0, stream>>>(wrap, size);
}

#define NVCV_TEST_INST_SET(PIXEL_TYPE) \
    template void DeviceSetOnes(nv::cv::cuda::Tensor2DWrap<PIXEL_TYPE> &, int2, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET

// ----------------- To allow testing device-side Tensor3DWrap -----------------

template<typename PixelType>
__global__ void Copy(cuda::Tensor3DWrap<PixelType> dst, cuda::Tensor3DWrap<const PixelType> src)
{
    int3 coord = cuda::StaticCast<int>(threadIdx);
    dst[coord] = src[coord];
}

template<typename PixelType, int N, int H, int W>
void DeviceUseTensor3DWrap(PackedTensor<PixelType, N, H, W> &hGold)
{
    PixelType *dInput;
    PixelType *dTest;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&dInput, N * H * W * sizeof(PixelType)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&dTest, N * H * W * sizeof(PixelType)));

    cuda::Tensor3DWrap<const PixelType> src(dInput, hGold.pitchBytes1, hGold.pitchBytes2);
    cuda::Tensor3DWrap<PixelType>       dst(dTest, hGold.pitchBytes1, hGold.pitchBytes2);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(dInput, hGold.data(), N * H * W * sizeof(PixelType), cudaMemcpyHostToDevice));

    Copy<<<1, dim3(W, H, N)>>>(dst, src);

    ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

    PackedTensor<PixelType, N, H, W> hTest;

    ASSERT_EQ(cudaSuccess, cudaMemcpy(hTest.data(), dTest, N * H * W * sizeof(PixelType), cudaMemcpyDeviceToHost));

    ASSERT_EQ(cudaSuccess, cudaFree(dInput));
    ASSERT_EQ(cudaSuccess, cudaFree(dTest));

    EXPECT_EQ(hTest, hGold);
}

// Need to instantiate each test on TestMem2DPackedWrap, making sure not to use const types

#define NVCV_TEST_INST_USE(PIXEL_TYPE, N, H, W) template void DeviceUseTensor3DWrap(PackedTensor<PIXEL_TYPE, N, H, W> &)

NVCV_TEST_INST_USE(int, 1, 2, 2);
NVCV_TEST_INST_USE(short3, 2, 2, 1);
NVCV_TEST_INST_USE(float1, 2, 2, 2);
NVCV_TEST_INST_USE(uchar4, 3, 3, 1);

#undef NVCV_TEST_INST_USE

template<typename PixelType>
__global__ void SetOnes(nv::cv::cuda::Tensor3DWrap<PixelType> dst, int3 size)
{
    int3 coord = cuda::StaticCast<int>(blockIdx * blockDim + threadIdx);

    if (coord.x >= size.x || coord.y >= size.y || coord.z >= size.z)
    {
        return;
    }

    dst[coord] = cuda::SetAll<PixelType>(1);
}

template<typename PixelType>
void DeviceSetOnes(nv::cv::cuda::Tensor3DWrap<PixelType> &wrap, int3 size, cudaStream_t &stream)
{
    dim3 block{32, 2, 2};
    dim3 grid{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y, (size.z + block.z - 1) / block.z};

    SetOnes<<<grid, block, 0, stream>>>(wrap, size);
}

#define NVCV_TEST_INST_SET(PIXEL_TYPE) \
    template void DeviceSetOnes(nv::cv::cuda::Tensor3DWrap<PIXEL_TYPE> &, int3, cudaStream_t &)

NVCV_TEST_INST_SET(int);
NVCV_TEST_INST_SET(uchar1);
NVCV_TEST_INST_SET(short2);
NVCV_TEST_INST_SET(uchar3);
NVCV_TEST_INST_SET(uchar4);
NVCV_TEST_INST_SET(float3);
NVCV_TEST_INST_SET(float4);

#undef NVCV_TEST_INST_SET
