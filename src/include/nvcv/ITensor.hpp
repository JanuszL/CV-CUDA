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

#include "PixelType.hpp"
#include "Tensor.h"
#include "TensorData.hpp"
#include "TensorLayout.hpp"
#include "TensorShape.hpp"
#include "detail/Optional.hpp"

namespace nv { namespace cv {

class ITensor
{
public:
    virtual ~ITensor() = default;

    NVCVTensorHandle handle() const;

    int          ndim() const;
    TensorShape  shape() const;
    PixelType    dtype() const;
    TensorLayout layout() const;

    const ITensorData *exportData() const;

private:
    virtual NVCVTensorHandle doGetHandle() const = 0;

    mutable detail::Optional<TensorDataPitchDevice> m_cacheData;
};

}} // namespace nv::cv

#include "detail/ITensorImpl.hpp"

#endif // NVCV_ITENSOR_HPP
