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

#ifndef NVCV_IMAGEBATCHDATA_IMPL_HPP
#define NVCV_IMAGEBATCHDATA_IMPL_HPP

#ifndef NVCV_IMAGEBATCHDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// ImageBatchVarShapeDataPitchDevice implementation -----------------------
inline ImageBatchVarShapeDataPitchDevice::ImageBatchVarShapeDataPitchDevice(int32_t numImages, const Buffer &buffer)
{
    NVCVImageBatchData &data = this->cdata();

    data.numImages            = numImages;
    data.bufferType           = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_PITCH_DEVICE;
    data.buffer.varShapePitch = buffer;
}

}} // namespace nv::cv

#endif // NVCV_IMAGEBATCHDATA_IMPL_HPP
