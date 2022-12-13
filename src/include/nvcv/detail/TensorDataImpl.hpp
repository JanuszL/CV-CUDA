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

#ifndef NVCV_TENSORDATA_IMPL_HPP
#define NVCV_TENSORDATA_IMPL_HPP

#ifndef NVCV_TENSORDATA_HPP
#    error "You must not include this header directly"
#endif

#include <algorithm>

namespace nv { namespace cv {

// TensorDataPitchDevice implementation -----------------------

inline TensorDataPitchDevice::TensorDataPitchDevice(const TensorShape &tshape, const PixelType &dtype,
                                                    const Buffer &buffer)
{
    NVCVTensorData &data = this->cdata();

    std::copy(tshape.shape().begin(), tshape.shape().end(), data.shape);
    data.ndim   = tshape.ndim();
    data.dtype  = dtype;
    data.layout = tshape.layout();

    data.bufferType   = NVCV_TENSOR_BUFFER_PITCH_DEVICE;
    data.buffer.pitch = buffer;
}

inline TensorDataPitchDevice::TensorDataPitchDevice(const NVCVTensorData &data)
    : ITensorDataPitchDevice(data)
{
}

}} // namespace nv::cv

#endif // NVCV_TENSORDATA_IMPL_HPP
