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

/**
 * @file TensorWrap.hpp
 *
 * @brief Defines N-D tensor wrapper with N pitches in bytes divided in compile- and run-time pitches.
 */

#ifndef NVCV_CUDA_TENSOR_WRAP_HPP
#define NVCV_CUDA_TENSOR_WRAP_HPP

#include "TypeTraits.hpp" // for HasTypeTraits, etc.

#include <nvcv/IImageData.hpp>  // for IImageDataPitchDevice, etc.
#include <nvcv/ITensorData.hpp> // for ITensorDataPitchDevice, etc.

#include <utility>

namespace nv::cv::cuda {

/**
 * @brief Tensor wrapper class used to wrap an N-D tensor with 0 to N compile-time pitches in bytes.
 *
 * @detail TensorWrap is a wrapper of an N-D tensor with 0 to N compile-time pitches.  Each pitch represents the
 * offset in bytes as a compile-time parameter that will be applied from the first (slowest changing) dimension to
 * the last (fastest changing) dimension of the tensor, in that order.  Each dimension with run-time pitch is
 * specified as -1 in the corresponding template argument.  For example, in the code below a wrap is defined for an
 * NHWC 4-D tensor where each batch image in N has a run-time image pitch (first -1 in template argument), and each
 * row in H has a run-time row pitch (second -1), a pixel in W has a compile-time constant pitch as the size of the
 * pixel type and a channel in C has also a compile-time constant pitch as the size of the channel type.  Elements
 * of type \p T are accessed via operator [] using an intD argument where D is the number of dimensions of the
 * tensor.  They can also be accessed via pointer using the \p ptr method with up to D integer arguments.
 *
 * @defgroup NVCV_CPP_CUDATOOLS_TENSORWRAP Tensor Wrapper classes
 * @{
 *
 * @code
 * using PixelType = ...;
 * using ChannelType = BaseType<PixelType>;
 * using Tensor = TensorWrap<PixelType, -1, -1, sizeof(PixelType), sizeof(ChannelType)>;
 * void *imageData = ...;
 * int planePitchBytes = ...;
 * int rowPitchBytes = ...;
 * Tensor tensor(imageData, planePitchBytes, rowPitchBytes);
 * @endcode
 *
 * @tparam T Type (it can be const) of each element inside the tensor wrapper.
 * @tparam Pitches Each compile-time (use -1 for run-time) pitch in bytes from first to last dimension.
 */
template<typename T, int... Pitches>
class TensorWrap;

template<typename T, int... Pitches>
class TensorWrap<const T, Pitches...>
{
    static_assert(HasTypeTraits<T>, "TensorWrap<T> can only be used if T has type traits");

public:
    using ValueType = const T;

    static constexpr int kNumDimensions   = sizeof...(Pitches);
    static constexpr int kVariablePitches = ((Pitches == -1) + ...);
    static constexpr int kConstantPitches = kNumDimensions - kVariablePitches;

    TensorWrap() = default;

    /**
     * @brief Constructs a constant TensorWrap by wrapping a const \p data pointer argument.
     *
     * @param[in] data Pointer to the data that will be wrapped
     * @param[in] pitchBytes0..D Each run-time pitch in bytes from first to last dimension
     */
    template<typename... Args>
    explicit __host__ __device__ TensorWrap(const void *data, Args... pitchBytes)
        : m_data(data)
        , m_pitchBytes{std::forward<int>(pitchBytes)...}
    {
        static_assert(std::conjunction_v<std::is_same<int, Args>...>);
        static_assert(sizeof...(Args) == kVariablePitches);
    }

    /**
     * @brief Constructs a constant TensorWrap by wrapping an \p image argument.
     *
     * @param[in] image Image reference to the image that will be wrapped
     */
    __host__ TensorWrap(const IImageDataPitchDevice &image)
    {
        static_assert(kVariablePitches == 1 && kNumDimensions == 2);

        m_data = reinterpret_cast<const void *>(image.plane(0).buffer);

        m_pitchBytes[0] = image.plane(0).pitchBytes;
    }

    /**
     * @brief Constructs a constant TensorWrap by wrapping a \p tensor argument.
     *
     * @param[in] tensor Tensor reference to the tensor that will be wrapped
     */
    __host__ TensorWrap(const ITensorDataPitchDevice &tensor)
    {
        m_data = reinterpret_cast<const void *>(tensor.data());

#pragma unroll
        for (int i = 0; i < kVariablePitches; ++i)
        {
            assert(tensor.pitchBytes(i) <= TypeTraits<int>::max);

            m_pitchBytes[i] = tensor.pitchBytes(i);
        }
    }

    /**
     * @brief Get run-time pitch in bytes.
     *
     * @return The const array (as a pointer) containing run-time pitches in bytes.
     */
    __host__ __device__ const int *pitchBytes() const
    {
        return m_pitchBytes;
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] c 1D coordinate (x first dimension) to be accessed
     *
     * @return Accessed const reference
     */
    inline const __host__ __device__ T &operator[](int1 c) const
    {
        return *doGetPtr(c.x);
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] c 2D coordinates (y first and x second dimension) to be accessed
     *
     * @return Accessed const reference
     */
    inline const __host__ __device__ T &operator[](int2 c) const
    {
        return *doGetPtr(c.y, c.x);
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] c 3D coordinates (z first, y second and x third dimension) to be accessed
     *
     * @return Accessed const reference
     */
    inline const __host__ __device__ T &operator[](int3 c) const
    {
        return *doGetPtr(c.z, c.y, c.x);
    }

    /**
     * @brief Subscript operator for read-only access.
     *
     * @param[in] c 4D coordinates (w first, z second, y third, and x fourth dimension) to be accessed
     *
     * @return Accessed const reference
     */
    inline const __host__ __device__ T &operator[](int4 c) const
    {
        return *doGetPtr(c.w, c.z, c.y, c.x);
    }

    /**
     * @brief Get a read-only proxy (as pointer) at the Dth dimension.
     *
     * @param[in] c0..D Each coordinate from first to last dimension
     *
     * @return The const pointer to the beginning of the Dth dimension
     */
    template<typename... Args>
    inline const __host__ __device__ T *ptr(Args... c) const
    {
        return doGetPtr(c...);
    }

protected:
    template<typename... Args>
    inline const __host__ __device__ T *doGetPtr(Args... c) const
    {
        static_assert(std::conjunction_v<std::is_same<int, Args>...>);
        static_assert(sizeof...(Args) <= kNumDimensions);

        constexpr int kArgSize      = sizeof...(Args);
        constexpr int kVarSize      = kArgSize < kVariablePitches ? kArgSize : kVariablePitches;
        constexpr int kDimSize      = kArgSize < kNumDimensions ? kArgSize : kNumDimensions;
        constexpr int kPitchBytes[] = {std::forward<int>(Pitches)...};

        int coords[] = {std::forward<int>(c)...};

        // Computing offset first potentially postpones or avoids 64-bit math during addressing
        int offset = 0;
#pragma unroll
        for (int i = 0; i < kVarSize; ++i)
        {
            offset += coords[i] * m_pitchBytes[i];
        }
#pragma unroll
        for (int i = kVariablePitches; i < kDimSize; ++i)
        {
            offset += coords[i] * kPitchBytes[i];
        }

        return reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(m_data) + offset);
    }

private:
    const void *m_data                         = nullptr;
    int         m_pitchBytes[kVariablePitches] = {};
};

template<typename T, int... Pitches>
class TensorWrap : public TensorWrap<const T, Pitches...>
{
    using Base = TensorWrap<const T, Pitches...>;

public:
    using ValueType = T;

    using Base::kConstantPitches;
    using Base::kNumDimensions;
    using Base::kVariablePitches;

    TensorWrap() = default;

    /**
     * @brief Constructs a TensorWrap by wrapping a \p data pointer argument.
     *
     * @param[in] data Pointer to the data that will be wrapped
     * @param[in] pitchBytes0..N Each run-time pitch in bytes from first to last dimension
     */
    template<typename... Args>
    explicit __host__ __device__ TensorWrap(void *data, Args... pitchBytes)
        : Base(data, pitchBytes...)
    {
    }

    /**
     * @brief Constructs a TensorWrap by wrapping an \p image argument.
     *
     * @param[in] image Image reference to the image that will be wrapped
     */
    __host__ TensorWrap(const IImageDataPitchDevice &image)
        : Base(image)
    {
    }

    /**
     * @brief Constructs a TensorWrap by wrapping a \p tensor argument.
     *
     * @param[in] tensor Tensor reference to the tensor that will be wrapped
     */
    __host__ TensorWrap(const ITensorDataPitchDevice &tensor)
        : Base(tensor)
    {
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] c 1D coordinate (x first dimension) to be accessed
     *
     * @return Accessed reference
     */
    inline __host__ __device__ T &operator[](int1 c) const
    {
        return *doGetPtr(c.x);
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] c 2D coordinates (y first and x second dimension) to be accessed
     *
     * @return Accessed reference
     */
    inline __host__ __device__ T &operator[](int2 c) const
    {
        return *doGetPtr(c.y, c.x);
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] c 3D coordinates (z first, y second and x third dimension) to be accessed
     *
     * @return Accessed reference
     */
    inline __host__ __device__ T &operator[](int3 c) const
    {
        return *doGetPtr(c.z, c.y, c.x);
    }

    /**
     * @brief Subscript operator for read-and-write access.
     *
     * @param[in] c 4D coordinates (w first, z second, y third, and x fourth dimension) to be accessed
     *
     * @return Accessed reference
     */
    inline __host__ __device__ T &operator[](int4 c) const
    {
        return *doGetPtr(c.w, c.z, c.y, c.x);
    }

    /**
     * @brief Get a read-and-write proxy (as pointer) at the Dth dimension.
     *
     * @param[in] c0..D Each coordinate from first to last dimension
     *
     * @return The pointer to the beginning of the Dth dimension
     */
    template<typename... Args>
    inline __host__ __device__ T *ptr(Args... c) const
    {
        return doGetPtr(c...);
    }

protected:
    template<typename... Args>
    inline __host__ __device__ T *doGetPtr(Args... c) const
    {
        // The const_cast here is the *only* place where it is used to remove the base pointer constness
        return const_cast<T *>(Base::doGetPtr(c...));
    }
};

/**
 * @brief Tensor 2D wrapper class.
 *
 * @detail Tensor2DWrap is a wrapper of a 2-D tensor, i.e. a matrix, with a fixed (compile-time) pitch in bytes for
 * the second dimension, columns of a matrix, as the size of type \p T.  The pitch for the second dimension, rows
 * of a matrix, is defined at run time, and can be seen as the matrix row pitch.  The operator [] used with int2
 * gets a reference to the y row (first dimension) and x column (second dimension).  The ptr method used with one
 * int i gets a pointer to the beginning of the i-th row.
 *
 * @tparam T Type (it can be const) of each element inside the 2D tensor wrapper.
 */
template<typename T>
using Tensor2DWrap = TensorWrap<T, -1, sizeof(T)>;

/**
 * @brief Tensor 3D wrapper class.
 *
 * @detail Tensor3DWrap is a wrapper of a 3-D tensor, e.g. a NHW tensor as N batches, H height and W width, with a
 * fixed (compile-time) pitch in bytes for the third dimension, columns of a tensor, as the size of type \p T.  The
 * pitch for the first dimension, slices of a tensor, is defined at run time, and can be seen as the tensor slice
 * pitch.  The pitch for the second dimension, rows of a tensor, is defined at run time, and can be seen as the
 * tensor row pitch.  The operator [] used with int3 gets a reference to the z batch (first dimension), y row
 * (second dimension) and x column (third dimension).  The ptr method used with one int b gets a pointer to the
 * beginning of the b-th batch.  The ptr method used with two int's b and i gets a pointer to the beginning of the
 * i-th row in the b-th batch.
 *
 * @tparam T Type (it can be const) of each element inside the 3D tensor wrapper.
 */
template<typename T>
using Tensor3DWrap = TensorWrap<T, -1, -1, sizeof(T)>;

/**
 * @brief Tensor 4D wrapper class.
 *
 * @detail Tensor4DWrap is a wrapper of a 4-D tensor, e.g. a NCHW tensor as N batches, C channel planes, H height
 * and W width, with a fixed (compile-time) pitch in bytes for the fourth dimension, columns of a tensor, as the
 * size of type \p T.  The pitch for the first dimension, slices of a tensor, is defined at run time, and can be
 * seen as the tensor slice pitch.  The pitch for the second dimension, planes of a tensor, is defined at run time,
 * and can be seen as the tensor plane pitch. The pitch for the third dimension, rows of a tensor, is defined at
 * run time, and can be seen as the tensor row pitch.  The operator [] used with int4 gets a reference to the w
 * batch (first dimension), z plane (second dimension), y row (third dimension) and x column (fourth dimension).
 * The ptr method used with one int b gets a pointer to the beginning of the b-th batch.  The ptr method used with
 * two int's b and c gets a pointer to the beginning of the c-th channel plane in the b-th batch.  The ptr method
 * used with three int's b, c and i gets a pointer to the beginning of the i-th row in the c-th channel plane in
 * the b-th batch.
 *
 * @tparam T Type (it can be const) of each element inside the 4D tensor wrapper.
 */
template<typename T>
using Tensor4DWrap = TensorWrap<T, -1, -1, -1, sizeof(T)>;

/**@}*/

} // namespace nv::cv::cuda

#endif // NVCV_CUDA_TENSOR_WRAP_HPP
