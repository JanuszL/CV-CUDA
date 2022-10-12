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

#include "TensorLayout.hpp"

#include "Exception.hpp"

namespace nv::cv::priv {

int32_t GetNDims(NVCVTensorLayout layout)
{
    switch (layout)
    {
    case NVCV_TENSOR_NCHW:
    case NVCV_TENSOR_NHWC:
        return 4;
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid tensor layout: " << layout;
}

} // namespace nv::cv::priv
