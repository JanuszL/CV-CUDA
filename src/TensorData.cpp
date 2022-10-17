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
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/TensorData.hpp>
#include <private/core/TensorLayout.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorLayoutGetNDims, (NVCVTensorLayout layout, int32_t *ndims))
{
    return priv::ProtectCall(
        [&]
        {
            if (ndims == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "ndims output must not be NULL");
            }

            *ndims = priv::GetNDims(layout);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorDataPitchDeviceFillForImages,
                (NVCVTensorData * data, NVCVImageFormat format, int32_t numImages, int32_t imgWidth, int32_t imgHeight,
                 void *mem, const int64_t *pitchBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "ndims output must not be NULL");
            }

            priv::FillTensorData(*data, priv::ImageFormat{format}, numImages, {imgWidth, imgHeight}, mem, pitchBytes);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorDataPitchDeviceFillDimsNCHW,
                (NVCVTensorData * data, NVCVImageFormat format, int32_t nbatch, int32_t channels, int32_t height,
                 int32_t width, void *mem, const int64_t *pitchBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (data == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "ndims output must not be NULL");
            }

            priv::FillTensorData(*data, priv::ImageFormat{format}, priv::DimsNCHW{nbatch, channels, height, width}, mem,
                                 pitchBytes);
        });
}
