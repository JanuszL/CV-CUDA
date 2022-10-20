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

#ifndef NVCV_ITENSOR_HPP
#define NVCV_ITENSOR_HPP

#include "Dims.hpp"
#include "ITensorData.hpp"
#include "Image.hpp"
#include "Tensor.h"
#include "detail/CudaFwd.h"
#include "detail/Optional.hpp"

#include <array>
#include <cassert>
#include <functional>
#include <iterator>

namespace nv { namespace cv {

class ITensor
{
public:
    virtual ~ITensor() = default;

    NVCVTensorHandle handle() const;

    int          ndim() const;
    Shape        shape() const;
    PixelType    dtype() const;
    TensorLayout layout() const;

    IAllocator &alloc() const;

    const ITensorData *exportData() const;

private:
    virtual NVCVTensorHandle doGetHandle() const = 0;

    virtual int          doGetNumDim() const   = 0;
    virtual TensorLayout doGetLayout() const   = 0;
    virtual Shape        doGetShape() const    = 0;
    virtual PixelType    doGetDataType() const = 0;

    virtual IAllocator &doGetAlloc() const = 0;

    virtual const ITensorData *doExportData() const = 0;
};

// Implementation

inline NVCVTensorHandle ITensor::handle() const
{
    return doGetHandle();
}

inline Shape ITensor::shape() const
{
    return doGetShape();
}

inline int ITensor::ndim() const
{
    return doGetNumDim();
}

inline TensorLayout ITensor::layout() const
{
    return doGetLayout();
}

inline PixelType ITensor::dtype() const
{
    return doGetDataType();
}

inline IAllocator &ITensor::alloc() const
{
    return doGetAlloc();
}

inline const ITensorData *ITensor::exportData() const
{
    return doExportData();
}

}} // namespace nv::cv

#endif // NVCV_ITENSOR_HPP
