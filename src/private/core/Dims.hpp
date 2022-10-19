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

#ifndef NVCV_PRIV_DIMS_HPP
#define NVCV_PRIV_DIMS_HPP

#include <nvcv/Tensor.h>
#include <util/Dims.hpp>

#include <array>

namespace nv::cv::priv {

using Shape = std::array<int32_t, NVCV_TENSOR_MAX_NDIM>;

using util::DimsNCHW;

DimsNCHW ToNCHW(const Shape &shape, NVCVTensorLayout layout);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_DIMS_HPP
