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

#include "TensorData.hpp"

#include "Exception.hpp"
#include "TensorLayout.hpp"

namespace nv::cv::priv {

void ValidateImageFormatForTensor(ImageFormat fmt) {}

NVCVTensorLayout GetTensorLayoutFor(ImageFormat fmt, int nbatches)
{
    (void)nbatches;

    int nplanes = fmt.numPlanes();

    if (nplanes == 1)
    {
        return NVCV_TENSOR_NHWC;
    }
    else if (nplanes == fmt.numChannels())
    {
        return NVCV_TENSOR_NCHW;
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format cannot be semi-planar, but it is: " << fmt;
    }
}

} // namespace nv::cv::priv
