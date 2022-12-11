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

#ifndef NVCV_ITENSORDATA_HPP
#define NVCV_ITENSORDATA_HPP

#include "PixelType.hpp"
#include "TensorData.h"
#include "TensorShape.hpp"
#include "detail/Optional.hpp"

namespace nv { namespace cv {

// Interface hierarchy of tensor contents
class ITensorData
{
public:
    virtual ~ITensorData() = 0;

    int                         ndim() const;
    const TensorShape          &shape() const;
    const TensorShape::DimType &shape(int d) const;

    const TensorLayout &layout() const;

    PixelType dtype() const;

    const NVCVTensorData &cdata() const;

protected:
    ITensorData() = default;
    ITensorData(const NVCVTensorData &data);

    NVCVTensorData &cdata();

private:
    NVCVTensorData                        m_data;
    mutable detail::Optional<TensorShape> m_cacheShape;
};

class ITensorDataPitch : public ITensorData
{
public:
    virtual ~ITensorDataPitch() = 0;

    void *data() const;

    const int64_t &pitchBytes(int d) const;

protected:
    using ITensorData::ITensorData;
};

class ITensorDataPitchDevice : public ITensorDataPitch
{
public:
    virtual ~ITensorDataPitchDevice() = 0;

protected:
    using ITensorDataPitch::ITensorDataPitch;
};

}} // namespace nv::cv

#include "detail/ITensorDataImpl.hpp"

#endif // NVCV_ITENSORDATA_HPP
