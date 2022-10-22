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
#include "TensorShape.hpp"

namespace nv { namespace cv {

// TensorDataPitchDevice definition -----------------------
class TensorDataPitchDevice final : public ITensorDataPitchDevice
{
public:
    using Buffer = NVCVTensorBufferPitch;

    explicit TensorDataPitchDevice(const TensorShape &shape, const PixelType &dtype, const Buffer &data);

private:
    NVCVTensorData m_data;

    int                         doGetNumDim() const override;
    TensorShape                 doGetShape() const override;
    const TensorShape::DimType &doGetShapeDim(int d) const override;
    DimsNCHW                    doGetDims() const override;
    TensorLayout                doGetLayout() const override;

    int32_t doGetNumPlanes() const override;
    int32_t doGetNumImages() const override;

    PixelType doGetPixelType() const override;

    const NVCVTensorData &doGetConstCData() const override;
    NVCVTensorData       &doGetCData() override;

    void *doGetData() const override;

    const int64_t &doGetPitchBytes(int d) const override;

    int64_t doGetImagePitchBytes() const override;
    int64_t doGetPlanePitchBytes() const override;
    int64_t doGetRowPitchBytes() const override;
    int64_t doGetColPitchBytes() const override;

    void *doGetImageBuffer(int n) const override;
    void *doGetImagePlaneBuffer(int n, int p) const override;
};

// TensorDataPitchDevice implementation -----------------------
inline TensorDataPitchDevice::TensorDataPitchDevice(const TensorShape &tshape, const PixelType &dtype,
                                                    const Buffer &data)
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

inline TensorShape TensorDataPitchDevice::doGetShape() const
{
    return TensorShape(m_data.shape, m_data.ndim, m_data.layout);
}

inline DimsNCHW TensorDataPitchDevice::doGetDims() const
{
    TensorShape::ShapeType nchw(4);
    detail::CheckThrow(nvcvTensorShapePermute(m_data.layout, m_data.shape, NVCV_TENSOR_NCHW, &nchw[0]));
    return {static_cast<int>(nchw[0]), static_cast<int>(nchw[1]), static_cast<int>(nchw[2]), static_cast<int>(nchw[3])};
}

inline int32_t TensorDataPitchDevice::doGetNumPlanes() const
{
    if (nvcvTensorLayoutCompare(m_data.layout, NVCV_TENSOR_NCHW) == 0)
    {
        return m_data.shape[1];
    }
    else if (nvcvTensorLayoutCompare(m_data.layout, NVCV_TENSOR_NHWC) == 0)
    {
        return 1;
    }
    else
    {
        assert(!"Invalid tensor layout");
        return -1;
    }
}

inline int32_t TensorDataPitchDevice::doGetNumImages() const
{
    return m_data.shape[0];
}

inline TensorLayout TensorDataPitchDevice::doGetLayout() const
{
    TensorLayout layout;
    static_assert(sizeof(layout) == sizeof(m_data.layout));
    // std::bitcast
    memcpy(&layout, &m_data.layout, sizeof(layout));
    return layout;
}

inline PixelType TensorDataPitchDevice::doGetPixelType() const
{
    return static_cast<PixelType>(m_data.dtype);
}

inline const NVCVTensorData &TensorDataPitchDevice::doGetConstCData() const
{
    return m_data;
}

inline NVCVTensorData &TensorDataPitchDevice::doGetCData()
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
    if (nvcvTensorLayoutCompare(m_data.layout, NVCV_TENSOR_NCHW) == 0)
    {
        return m_data.buffer.pitch.pitchBytes[2];
    }
    else if (nvcvTensorLayoutCompare(m_data.layout, NVCV_TENSOR_NHWC) == 0)
    {
        return m_data.buffer.pitch.pitchBytes[1];
    }
    else
    {
        return 0;
    }
}

inline int64_t TensorDataPitchDevice::doGetColPitchBytes() const
{
    if (nvcvTensorLayoutCompare(m_data.layout, NVCV_TENSOR_NCHW) == 0)
    {
        return m_data.buffer.pitch.pitchBytes[3];
    }
    else if (nvcvTensorLayoutCompare(m_data.layout, NVCV_TENSOR_NHWC) == 0)
    {
        return m_data.buffer.pitch.pitchBytes[2];
    }
    else
    {
        return 0;
    }
}

inline void *TensorDataPitchDevice::doGetImageBuffer(int n) const
{
    assert(n >= 0 && n < m_data.shape[0]);

    return reinterpret_cast<std::byte *>(m_data.buffer.pitch.data) + m_data.buffer.pitch.pitchBytes[0] * n;
}

inline void *TensorDataPitchDevice::doGetImagePlaneBuffer(int n, int p) const
{
    assert(p >= 0 && p < m_data.shape[1]);

    return reinterpret_cast<std::byte *>(doGetImageBuffer(n)) + m_data.buffer.pitch.pitchBytes[1] * p;
}

}} // namespace nv::cv

#endif // NVCV_TENSORDATA_HPP
