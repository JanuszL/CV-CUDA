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

#include <nvcv/TensorData.h>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorShapeInfo.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/TensorShape.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorShapePermute,
                (NVCVTensorLayout srcLayout, const int64_t *srcShape, NVCVTensorLayout dstLayout, int64_t *dstShape))
{
    return priv::ProtectCall([&] { priv::PermuteShape(srcLayout, srcShape, dstLayout, dstShape); });
}
