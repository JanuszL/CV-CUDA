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

#ifndef NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP
#define NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP

#include <cuda_runtime.h>           // for int2, etc.
#include <nvcv/cuda/TensorWrap.hpp> // the object of this test
#include <nvcv/cuda/TypeTraits.hpp> // for NumElements, etc.

#include <array> // for std::array, etc.

// clang-format off

template<typename T, int H, int W>
struct PackedImage { // PackedImage extends std::array in two dimensions
    using value_type = T;
    static constexpr int pitchBytes = W * sizeof(T), height = H, width = W;
    std::array<T, H*W> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedImage<T, H, W> &that) const { return m_data == that.m_data; }
};

template<typename T, int N, int H, int W>
struct PackedTensor { // PackedTensor extends std::array in three dimensions
    using value_type = T;
    static constexpr int C = nv::cv::cuda::NumElements<T>;
    static constexpr int batches = N, height = H, width = W, channels = C;
    static constexpr int pitchBytes4 = sizeof(nv::cv::cuda::BaseType<T>);
    static constexpr int pitchBytes3 = C * pitchBytes4;
    static constexpr int pitchBytes2 = W * pitchBytes3;
    static constexpr int pitchBytes1 = H * pitchBytes2;

    std::array<T, N*H*W> m_data;
    T *data() { return m_data.data(); }
    const T *data() const { return m_data.data(); }
    const T& operator[](int i) const { return m_data[i]; }
    bool operator==(const PackedTensor<T, N, H, W> &that) const { return m_data == that.m_data; }
};

// clang-format on

template<typename PixelType, int H, int W>
void DeviceUseTensor2DWrap(PackedImage<PixelType, H, W> &);

template<typename PixelType, int N, int H, int W>
void DeviceUseTensor3DWrap(PackedTensor<PixelType, N, H, W> &);

template<typename PixelType>
void DeviceSetOnes(nv::cv::cuda::Tensor2DWrap<PixelType> &, int2, cudaStream_t &);

template<typename PixelType>
void DeviceSetOnes(nv::cv::cuda::Tensor3DWrap<PixelType> &, int3, cudaStream_t &);

#endif // NVCV_TESTS_DEVICE_TENSOR_WRAP_HPP
