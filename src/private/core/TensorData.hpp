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

#ifndef NVCV_PRIV_TENSORDATA_HPP
#define NVCV_PRIV_TENSORDATA_HPP

#include "Size.hpp"

#include <fmt/ImageFormat.hpp>
#include <nvcv/TensorData.h>

namespace nv::cv::priv {

NVCVTensorLayout GetTensorLayoutFor(ImageFormat fmt, int nbatches);

void ValidateImageFormatForTensor(ImageFormat fmt);

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TENSORDATA_HPP
