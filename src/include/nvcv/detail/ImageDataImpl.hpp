/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_IMAGEDATA_IMPL_HPP
#define NVCV_IMAGEDATA_IMPL_HPP

#ifndef NVCV_IMAGEDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// ImageDataCudaArray implementation -----------------------

inline ImageDataCudaArray::ImageDataCudaArray(ImageFormat format, const Buffer &buffer)
{
    NVCVImageData &data = this->cdata();

    data.format           = format;
    data.bufferType       = NVCV_IMAGE_BUFFER_CUDA_ARRAY;
    data.buffer.cudaarray = buffer;
}

inline ImageDataCudaArray::ImageDataCudaArray(const NVCVImageData &data)
    : IImageDataCudaArray(data)
{
    if (data.bufferType != NVCV_IMAGE_BUFFER_CUDA_ARRAY)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "Image buffer format must be suitable for cuda arrays (block-linear)");
    }
}

// ImageDataPitchDevice implementation -----------------------

inline ImageDataPitchDevice::ImageDataPitchDevice(ImageFormat format, const Buffer &buffer)
{
    NVCVImageData &data = this->cdata();

    data.format       = format;
    data.bufferType   = NVCV_IMAGE_BUFFER_PITCH_DEVICE;
    data.buffer.pitch = buffer;
}

inline ImageDataPitchDevice::ImageDataPitchDevice(const NVCVImageData &data)
    : IImageDataPitchDevice(data)
{
    if (data.bufferType != NVCV_IMAGE_BUFFER_PITCH_DEVICE)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "Image buffer format must be suitable for pitch-linear CUDA-accessible data");
    }
}

// ImageDataPitchHost implementation -----------------------

inline ImageDataPitchHost::ImageDataPitchHost(ImageFormat format, const Buffer &buffer)
{
    NVCVImageData &data = this->cdata();

    data.format       = format;
    data.bufferType   = NVCV_IMAGE_BUFFER_PITCH_HOST;
    data.buffer.pitch = buffer;
}

inline ImageDataPitchHost::ImageDataPitchHost(const NVCVImageData &data)
    : IImageDataPitchHost(data)
{
    if (data.bufferType != NVCV_IMAGE_BUFFER_PITCH_DEVICE)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT,
                        "Image buffer format must be suitable for pitch-linear host-accessible data");
    }
}

}} // namespace nv::cv

#endif // NVCV_IMAGEDATA_IMPL_HPP
