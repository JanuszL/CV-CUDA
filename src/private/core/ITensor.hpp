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

#ifndef NVCV_PRIV_ITENSOR_HPP
#define NVCV_PRIV_ITENSOR_HPP

#include "Dims.hpp"
#include "ICoreObject.hpp"

#include <fmt/ImageFormat.hpp>
#include <nvcv/Tensor.h>

namespace nv::cv::priv {

class IAllocator;

class ITensor : public ICoreObjectHandle<ITensor, NVCVTensorHandle>
{
public:
    virtual int32_t        ndim() const  = 0;
    virtual const int32_t *shape() const = 0;

    virtual NVCVTensorLayout layout() const = 0;
    virtual DimsNCHW         dims() const   = 0;

    virtual PixelType dtype() const = 0;

    virtual IAllocator &alloc() const = 0;

    virtual void exportData(NVCVTensorData &data) const = 0;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_ITENSOR_HPP
