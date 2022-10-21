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

namespace nv { namespace cv {

// Interface hierarchy of tensor contents
class ITensorData
{
public:
    virtual ~ITensorData() = default;

    int                ndim() const;
    const TensorShape &shape() const;

    const TensorShape::DimType &shape(int d) const;

    const TensorLayout &layout() const;

    PixelType dtype() const;

    const NVCVTensorData &cdata() const;

private:
    // NVI idiom
    virtual int                         doGetNumDim() const        = 0;
    virtual const TensorShape          &doGetShape() const         = 0;
    virtual const TensorShape::DimType &doGetShapeDim(int d) const = 0;

    virtual PixelType doGetPixelType() const = 0;

    virtual const NVCVTensorData &doGetCData() const = 0;
};

class ITensorDataPitch : public ITensorData
{
public:
    void          *data() const;
    const int64_t &pitchBytes(int d) const;

private:
    virtual void *doGetData() const = 0;

    virtual const int64_t &doGetPitchBytes(int d) const = 0;
};

class ITensorDataPitchDevice : public ITensorDataPitch
{
};

// Implementation - ITensorData
inline int ITensorData::ndim() const
{
    int r = doGetNumDim();
    assert(1 <= r && r <= NVCV_TENSOR_MAX_NDIM);
    return r;
}

inline const TensorShape &ITensorData::shape() const
{
    return doGetShape();
}

inline auto ITensorData::shape(int d) const -> const TensorShape::DimType &
{
    return doGetShapeDim(d);
}

inline const TensorLayout &ITensorData::layout() const
{
    return doGetShape().layout();
}

inline PixelType ITensorData::dtype() const
{
    return doGetPixelType();
}

inline const NVCVTensorData &ITensorData::cdata() const
{
    return doGetCData();
}

// Implementation - ITensorDataPitch

inline void *ITensorDataPitch::data() const
{
    void *data = doGetData();
    assert(data != nullptr);
    return data;
}

inline const int64_t &ITensorDataPitch::pitchBytes(int d) const
{
    const int64_t &p = doGetPitchBytes(d);
    assert(p > 0 && "Post-condition failed");
    return p;
}

}} // namespace nv::cv

#endif // NVCV_ITENSORDATA_HPP
