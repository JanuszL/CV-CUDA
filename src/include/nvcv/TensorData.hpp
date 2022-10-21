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

#include "ITensorData.hpp"
#include "TensorShape.hpp"

namespace nv { namespace cv {

// TensorDataPitchDevice definition -----------------------
class TensorDataPitchDevice final : public ITensorDataPitchDevice
{
public:
    using Buffer = NVCVTensorBufferPitch;

    explicit TensorDataPitchDevice(const TensorShape &shape, const PixelType &dtype, const Buffer &data);

private:
    NVCVTensorData      m_data;
    mutable TensorShape m_cacheShape;

    int                         doGetNumDim() const override;
    const TensorShape          &doGetShape() const override;
    const TensorShape::DimType &doGetShapeDim(int d) const override;

    PixelType doGetPixelType() const override;

    const NVCVTensorData &doGetCData() const override;

    void *doGetData() const override;

    const int64_t &doGetPitchBytes(int d) const override;
};

// TensorDataPitchDevice implementation -----------------------
inline TensorDataPitchDevice::TensorDataPitchDevice(const TensorShape &tshape, const PixelType &dtype,
                                                    const Buffer &data)
    : m_cacheShape(tshape)
{
    std::copy(tshape.shape().begin(), tshape.shape().end(), m_data.shape);
    m_data.ndim   = tshape.ndim();
    m_data.dtype  = dtype;
    m_data.layout = tshape.layout();

    m_data.bufferType   = NVCV_TENSOR_BUFFER_PITCH_DEVICE;
    m_data.buffer.pitch = data;
}

inline int TensorDataPitchDevice::doGetNumDim() const
{
    return m_data.ndim;
}

inline const TensorShape::DimType &TensorDataPitchDevice::doGetShapeDim(int d) const
{
    return m_data.shape[d];
}

inline const TensorShape &TensorDataPitchDevice::doGetShape() const
{
    return m_cacheShape;
}

inline PixelType TensorDataPitchDevice::doGetPixelType() const
{
    return static_cast<PixelType>(m_data.dtype);
}

inline const NVCVTensorData &TensorDataPitchDevice::doGetCData() const
{
    return m_data;
}

inline void *TensorDataPitchDevice::doGetData() const
{
    return m_data.buffer.pitch.data;
}

inline const int64_t &TensorDataPitchDevice::doGetPitchBytes(int d) const
{
    return m_data.buffer.pitch.pitchBytes[d];
}

}} // namespace nv::cv

#endif // NVCV_TENSORDATA_HPP
