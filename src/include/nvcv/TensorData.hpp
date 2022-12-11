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
#include "PixelType.hpp"
#include "TensorShape.hpp"

namespace nv { namespace cv {

// TensorDataPitchDevice definition -----------------------

class TensorDataPitchDevice : public ITensorDataPitchDevice
{
public:
    using Buffer = NVCVTensorBufferPitch;

    explicit TensorDataPitchDevice(const TensorShape &shape, const PixelType &dtype, const Buffer &data);
    explicit TensorDataPitchDevice(const NVCVTensorData &data);
};

}} // namespace nv::cv

#include "detail/TensorDataImpl.hpp"

#endif // NVCV_TENSORDATA_HPP
