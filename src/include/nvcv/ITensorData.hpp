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

#include "Dims.hpp"
#include "PixelType.hpp"
#include "Shape.hpp"
#include "TensorData.h"

namespace nv { namespace cv {

enum TensorLayout
{
    NCHW = NVCV_TENSOR_NCHW,
    NHWC = NVCV_TENSOR_NHWC,
};

// Interface hierarchy of tensor contents
class ITensorData
{
public:
    virtual ~ITensorData() = default;

    int   ndim() const;
    Shape shape() const;

    const int32_t &shape(int d) const;

    DimsNCHW     dims() const;
    TensorLayout layout() const;

    PixelType dtype() const;

    int32_t numPlanes() const;
    int32_t numImages() const;

    const NVCVTensorData &cdata() const;
    NVCVTensorData       &cdata();

private:
    // NVI idiom
    virtual int                    doGetNumDim() const        = 0;
    virtual Shape                  doGetShape() const         = 0;
    virtual Shape::const_reference doGetShapeDim(int d) const = 0;
    virtual DimsNCHW               doGetDims() const          = 0;
    virtual TensorLayout           doGetLayout() const        = 0;

    virtual int32_t doGetNumPlanes() const = 0;
    virtual int32_t doGetNumImages() const = 0;

    virtual PixelType doGetPixelType() const = 0;

    virtual const NVCVTensorData &doGetConstCData() const = 0;
    virtual NVCVTensorData       &doGetCData()            = 0;
};

class ITensorDataPitch : public ITensorData
{
public:
    void          *mem() const;
    const int64_t &pitchBytes(int d) const;

    int64_t imgPitchBytes() const;
    int64_t rowPitchBytes() const;
    int64_t colPitchBytes() const;
    int64_t planePitchBytes() const;

    void *imgBuffer(int n) const;
    void *imgPlaneBuffer(int n, int p) const;

private:
    virtual void *doGetMemBuffer() const = 0;

    virtual const int64_t &doGetPitchBytes(int d) const = 0;

    virtual int64_t doGetImagePitchBytes() const = 0;
    virtual int64_t doGetPlanePitchBytes() const = 0;
    virtual int64_t doGetRowPitchBytes() const   = 0;
    virtual int64_t doGetColPitchBytes() const   = 0;

    virtual void *doGetImageBuffer(int n) const             = 0;
    virtual void *doGetImagePlaneBuffer(int n, int p) const = 0;
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

inline Shape ITensorData::shape() const
{
    return doGetShape();
}

inline const int32_t &ITensorData::shape(int d) const
{
    return doGetShapeDim(d);
}

inline DimsNCHW ITensorData::dims() const
{
    DimsNCHW d = doGetDims();
    assert(d.n >= 1 && d.c >= 1 && d.h >= 1 && d.w >= 1);
    return d;
}

inline int32_t ITensorData::numPlanes() const
{
    int32_t p = doGetNumPlanes();
    assert(p >= 1 && "Post-condition failed");
    return p;
}

inline int32_t ITensorData::numImages() const
{
    int32_t i = doGetNumImages();
    assert(i >= 1 && "Post-condition failed");
    return i;
}

inline TensorLayout ITensorData::layout() const
{
    return doGetLayout();
}

inline PixelType ITensorData::dtype() const
{
    return doGetPixelType();
}

inline const NVCVTensorData &ITensorData::cdata() const
{
    return doGetConstCData();
}

inline NVCVTensorData &ITensorData::cdata()
{
    return doGetCData();
}

// Implementation - ITensorDataPitch

inline void *ITensorDataPitch::mem() const
{
    void *mem = doGetMemBuffer();
    assert(mem != nullptr);
    return mem;
}

inline const int64_t &ITensorDataPitch::pitchBytes(int d) const
{
    const int64_t &p = doGetPitchBytes(d);
    assert(p > 0 && "Post-condition failed");
    return p;
}

inline void *ITensorDataPitch::imgBuffer(int n) const
{
    return doGetImageBuffer(n);
}

inline void *ITensorDataPitch::imgPlaneBuffer(int n, int p) const
{
    return doGetImagePlaneBuffer(n, p);
}

inline int64_t ITensorDataPitch::imgPitchBytes() const
{
    int64_t p = doGetImagePitchBytes();
    assert(p >= 1 && "Post-condition failed");
    return p;
}

inline int64_t ITensorDataPitch::planePitchBytes() const
{
    int64_t p = doGetPlanePitchBytes();
    assert(p >= 1 && "Post-condition failed");
    return p;
}

inline int64_t ITensorDataPitch::rowPitchBytes() const
{
    int64_t p = doGetRowPitchBytes();
    assert(p >= 1 && "Post-condition failed");
    return p;
}

inline int64_t ITensorDataPitch::colPitchBytes() const
{
    int64_t p = doGetColPitchBytes();
    assert(p >= 1 && "Post-condition failed");
    return p;
}

}} // namespace nv::cv

#endif // NVCV_ITENSORDATA_HPP
