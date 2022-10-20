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

    explicit TensorDataPitchDevice(const Buffer &data);

private:
    NVCVTensorData m_data;

    int                    doGetNumDim() const override;
    Shape                  doGetShape() const override;
    Shape::const_reference doGetShapeDim(int d) const override;
    DimsNCHW               doGetDims() const override;
    TensorLayout           doGetLayout() const override;

    int32_t doGetNumPlanes() const override;
    int32_t doGetNumImages() const override;

    PixelType doGetPixelType() const override;

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
inline TensorDataPitchDevice::TensorDataPitchDevice(const Buffer &data)
{
    m_data.bufferType   = NVCV_TENSOR_BUFFER_PITCH_DEVICE;
    m_data.buffer.pitch = data;
}

inline int TensorDataPitchDevice::doGetNumDim() const
{
    return m_data.buffer.pitch.ndim;
}

inline Shape::const_reference TensorDataPitchDevice::doGetShapeDim(int d) const
{
    return m_data.buffer.pitch.shape[d];
}

inline Shape TensorDataPitchDevice::doGetShape() const
{
    const NVCVTensorBufferPitch &pitch = m_data.buffer.pitch;

    return Shape(pitch.shape, pitch.shape + pitch.ndim);
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

inline PixelType TensorDataPitchDevice::doGetPixelType() const
{
    return static_cast<PixelType>(m_data.buffer.pitch.dtype);
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
