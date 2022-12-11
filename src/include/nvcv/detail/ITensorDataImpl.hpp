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

#ifndef NVCV_ITENSORDATA_IMPL_HPP
#define NVCV_ITENSORDATA_IMPL_HPP

#ifndef NVCV_ITENSORDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// Implementation - ITensorData -----------------------------

inline ITensorData::ITensorData(const NVCVTensorData &data)
    : m_data(data)
{
}

inline ITensorData::~ITensorData()
{
    // required dtor implementation
}

inline int ITensorData::ndim() const
{
    return this->cdata().ndim;
}

inline const TensorShape &ITensorData::shape() const
{
    if (!m_cacheShape)
    {
        const NVCVTensorData &data = this->cdata();
        m_cacheShape.emplace(data.shape, data.ndim, data.layout);
    }

    return *m_cacheShape;
}

inline auto ITensorData::shape(int d) const -> const TensorShape::DimType &
{
    const NVCVTensorData &data = this->cdata();

    if (d < 0 || d >= data.ndim)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Index of shape dimension %d is out of bounds [0;%d]", d,
                        data.ndim - 1);
    }
    return data.shape[d];
}

inline const TensorLayout &ITensorData::layout() const
{
    return this->shape().layout();
}

inline PixelType ITensorData::dtype() const
{
    const NVCVTensorData &data = this->cdata();
    return PixelType{data.dtype};
}

inline const NVCVTensorData &ITensorData::cdata() const
{
    return m_data;
}

inline NVCVTensorData &ITensorData::cdata()
{
    // data contents might be modified, must reset cache
    m_cacheShape.reset();
    return m_data;
}

// Implementation - ITensorDataPitch ----------------------------

inline ITensorDataPitch::~ITensorDataPitch()
{
    // required dtor implementation
}

inline void *ITensorDataPitch::data() const
{
    const NVCVTensorBufferPitch &buffer = this->cdata().buffer.pitch;
    return buffer.data;
}

inline const int64_t &ITensorDataPitch::pitchBytes(int d) const
{
    const NVCVTensorData &data = this->cdata();
    if (d < 0 || d >= data.ndim)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Index of pitch %d is out of bounds [0;%d]", d, data.ndim - 1);
    }

    return data.buffer.pitch.pitchBytes[d];
}

// Implementation - ITensorDataPitchDevice ----------------------------
inline ITensorDataPitchDevice::~ITensorDataPitchDevice()
{
    // required dtor implementation
}

}} // namespace nv::cv

#endif // NVCV_ITENSORDATA_IMPL_HPP
