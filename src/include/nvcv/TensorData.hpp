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

#ifndef NVCV_TENSORDATA_HPP
#define NVCV_TENSORDATA_HPP

#include "Dims.hpp"
#include "ITensorData.hpp"

namespace nv { namespace cv {

// TensorDataPitchDevice definition -----------------------
class TensorDataPitchDevice final : public ITensorDataPitchDevice
{
public:
    using Buffer = NVCVTensorBufferPitch;

    explicit TensorDataPitchDevice(ImageFormat format, const Buffer &data);

    explicit TensorDataPitchDevice(ImageFormat format, const DimsNCHW &dims, void *mem,
                                   const int64_t *pitchBytes = nullptr);

    explicit TensorDataPitchDevice(ImageFormat format, int numImages, const Size2D &size, void *mem,
                                   const int64_t *pitchBytes = nullptr);

private:
    NVCVTensorData m_data;

    int          doGetNumDim() const override;
    const Shape &doGetShape() const override;
    DimsNCHW     doGetDims() const override;
    TensorLayout doGetLayout() const override;

    int32_t doGetNumPlanes() const override;
    int32_t doGetNumImages() const override;

    ImageFormat doGetFormat() const override;

    const NVCVTensorData &doGetConstCData() const override;
    NVCVTensorData       &doGetCData() override;

    void *doGetMemBuffer() const override;

    const int64_t &doGetPitchBytes(int d) const override;

    int64_t doGetImagePitchBytes() const override;
    int64_t doGetPlanePitchBytes() const override;
    int64_t doGetRowPitchBytes() const override;
    int64_t doGetColPitchBytes() const override;

    void *doGetImageBuffer(int n) const override;
    void *doGetImagePlaneBuffer(int n, int p) const override;
};

// TensorDataPitchDevice implementation -----------------------
inline TensorDataPitchDevice::TensorDataPitchDevice(ImageFormat format, const Buffer &data)
{
    m_data.format       = format;
    m_data.bufferType   = NVCV_TENSOR_BUFFER_PITCH_DEVICE;
    m_data.buffer.pitch = data;
}

inline TensorDataPitchDevice::TensorDataPitchDevice(ImageFormat format, const DimsNCHW &dims, void *mem,
                                                    const int64_t *pitchBytes)
{
    detail::CheckThrow(
        nvcvTensorDataPitchDeviceFillDimsNCHW(&m_data, format, dims.n, dims.c, dims.h, dims.w, mem, pitchBytes));
}

inline TensorDataPitchDevice::TensorDataPitchDevice(ImageFormat format, int numImages, const Size2D &size, void *mem,
                                                    const int64_t *pitchBytes)
{
    detail::CheckThrow(
        nvcvTensorDataPitchDeviceFillForImages(&m_data, format, numImages, size.w, size.h, mem, pitchBytes));
}

inline int TensorDataPitchDevice::doGetNumDim() const
{
    int32_t ndim;
    detail::CheckThrow(nvcvTensorLayoutGetNumDim(m_data.buffer.pitch.layout, &ndim));
    return ndim;
}

inline const Shape &TensorDataPitchDevice::doGetShape() const
{
    static_assert(sizeof(Shape) / sizeof(Shape::value_type)
                  == sizeof(m_data.buffer.pitch.shape) / sizeof(m_data.buffer.pitch.shape[0]));
    static_assert(std::is_same<Shape::value_type, std::decay<decltype(m_data.buffer.pitch.shape[0])>::type>::value);

    // UB under stricter C++ rules, but fine in practice.
    return *reinterpret_cast<const Shape *>(m_data.buffer.pitch.shape);
}

inline DimsNCHW TensorDataPitchDevice::doGetDims() const
{
    const int32_t *shape = m_data.buffer.pitch.shape;

    switch (m_data.buffer.pitch.layout)
    {
    case NVCV_TENSOR_NCHW:
        return {shape[0], shape[1], shape[2], shape[3]};
    case NVCV_TENSOR_NHWC:
        return {shape[0], shape[3], shape[1], shape[2]};
    }
    assert(false && "Unknown tensor layout");
    return {};
}

inline int32_t TensorDataPitchDevice::doGetNumPlanes() const
{
    switch (m_data.buffer.pitch.layout)
    {
    case NVCV_TENSOR_NCHW:
        return m_data.buffer.pitch.shape[1];
    case NVCV_TENSOR_NHWC:
        return 1;
    }
    assert(!"Invalid tensor layout");
    // hopefully something will break real bad in release mode
    return -1;
}

inline int32_t TensorDataPitchDevice::doGetNumImages() const
{
    return m_data.buffer.pitch.shape[0];
}

inline TensorLayout TensorDataPitchDevice::doGetLayout() const
{
    return static_cast<TensorLayout>(m_data.buffer.pitch.layout);
}

inline ImageFormat TensorDataPitchDevice::doGetFormat() const
{
    return static_cast<ImageFormat>(m_data.format);
}

inline const NVCVTensorData &TensorDataPitchDevice::doGetConstCData() const
{
    return m_data;
}

inline NVCVTensorData &TensorDataPitchDevice::doGetCData()
{
    return m_data;
}

inline void *TensorDataPitchDevice::doGetMemBuffer() const
{
    return m_data.buffer.pitch.mem;
}

inline const int64_t &TensorDataPitchDevice::doGetPitchBytes(int d) const
{
    return m_data.buffer.pitch.pitchBytes[d];
}

inline int64_t TensorDataPitchDevice::doGetImagePitchBytes() const
{
    return m_data.buffer.pitch.pitchBytes[0];
}

inline int64_t TensorDataPitchDevice::doGetPlanePitchBytes() const
{
    return m_data.buffer.pitch.pitchBytes[1];
}

inline int64_t TensorDataPitchDevice::doGetRowPitchBytes() const
{
    switch (m_data.buffer.pitch.layout)
    {
    case NVCV_TENSOR_NCHW:
        return m_data.buffer.pitch.pitchBytes[2];
    case NVCV_TENSOR_NHWC:
        return m_data.buffer.pitch.pitchBytes[1];
    }
    assert(!"Invalid tensor layout");
    // hopefully something will break real bad in release mode
    return m_data.buffer.pitch.pitchBytes[0];
}

inline int64_t TensorDataPitchDevice::doGetColPitchBytes() const
{
    switch (m_data.buffer.pitch.layout)
    {
    case NVCV_TENSOR_NCHW:
        return m_data.buffer.pitch.pitchBytes[3];
    case NVCV_TENSOR_NHWC:
        return m_data.buffer.pitch.pitchBytes[2];
    }
    assert(!"Invalid tensor layout");
    // hopefully something will break real bad in release mode
    return m_data.buffer.pitch.pitchBytes[0];
}

inline void *TensorDataPitchDevice::doGetImageBuffer(int n) const
{
    assert(n >= 0 && n < m_data.buffer.pitch.shape[0]);

    return reinterpret_cast<std::byte *>(m_data.buffer.pitch.mem) + m_data.buffer.pitch.pitchBytes[0] * n;
}

inline void *TensorDataPitchDevice::doGetImagePlaneBuffer(int n, int p) const
{
    assert(p >= 0 && p < m_data.buffer.pitch.shape[1]);

    return reinterpret_cast<std::byte *>(doGetImageBuffer(n)) + m_data.buffer.pitch.pitchBytes[1] * p;
}

}} // namespace nv::cv

#endif // NVCV_TENSORDATA_HPP
