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
#include "detail/BaseFromMember.hpp"

namespace nv { namespace cv {

// TensorData definition -----------------------
class TensorDataWrap : public virtual ITensorData
{
public:
    explicit TensorDataWrap(const NVCVTensorData &data);

private:
    const NVCVTensorData &m_data;
    mutable TensorShape   m_cacheShape;

    int                         doGetNumDim() const override;
    const TensorShape          &doGetShape() const override;
    const TensorShape::DimType &doGetShapeDim(int d) const override;

    PixelType doGetPixelType() const override;

    const NVCVTensorData &doGetCData() const override;
};

// TensorDataPitchDevice definition -----------------------
class TensorDataPitchDevice final
    : public ITensorDataPitchDevice
    , private detail::BaseFromMember<NVCVTensorData>
    , private TensorDataWrap
{
public:
    using Buffer = NVCVTensorBufferPitch;

    explicit TensorDataPitchDevice(const TensorShape &shape, const PixelType &dtype, const Buffer &data);

private:
    using MemberTensorData = detail::BaseFromMember<NVCVTensorData>;

    void          *doGetData() const override;
    const int64_t &doGetPitchBytes(int d) const override;
};

// TensorDataWrap implementation -----------------------

inline TensorDataWrap::TensorDataWrap(const NVCVTensorData &data)
    : m_data(data)
{
    TensorShape::ShapeType shape(data.ndim);

    std::copy(m_data.shape, m_data.shape + m_data.ndim, shape.begin());

    m_cacheShape = TensorShape{std::move(shape), data.layout};
}

inline int TensorDataWrap::doGetNumDim() const
{
    return m_data.ndim;
}

inline const TensorShape::DimType &TensorDataWrap::doGetShapeDim(int d) const
{
    return m_data.shape[d];
}

inline const TensorShape &TensorDataWrap::doGetShape() const
{
    return m_cacheShape;
}

inline PixelType TensorDataWrap::doGetPixelType() const
{
    return static_cast<PixelType>(m_data.dtype);
}

inline const NVCVTensorData &TensorDataWrap::doGetCData() const
{
    return m_data;
}

// TensorDataPitchDevice implementation -----------------------
inline TensorDataPitchDevice::TensorDataPitchDevice(const TensorShape &tshape, const PixelType &dtype,
                                                    const Buffer &data)
    : MemberTensorData{[&]
                       {
                           NVCVTensorData tdata;

                           std::copy(tshape.shape().begin(), tshape.shape().end(), tdata.shape);
                           tdata.ndim   = tshape.ndim();
                           tdata.dtype  = dtype;
                           tdata.layout = tshape.layout();

                           tdata.bufferType   = NVCV_TENSOR_BUFFER_PITCH_DEVICE;
                           tdata.buffer.pitch = data;

                           return tdata;
                       }()}
    , TensorDataWrap(MemberTensorData::member)
{
}

inline void *TensorDataPitchDevice::doGetData() const
{
    return MemberTensorData::member.buffer.pitch.data;
}

inline const int64_t &TensorDataPitchDevice::doGetPitchBytes(int d) const
{
    return MemberTensorData::member.buffer.pitch.pitchBytes[d];
}

}} // namespace nv::cv

#endif // NVCV_TENSORDATA_HPP
