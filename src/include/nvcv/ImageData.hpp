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

#ifndef NVCV_IMAGEDATA_HPP
#define NVCV_IMAGEDATA_HPP

#include "IImageData.hpp"

namespace nv { namespace cv {

// ImageDataCudaArray definition -----------------------
class ImageDataCudaArray : public IImageDataCudaArray
{
public:
    using Buffer = NVCVImageBufferCudaArray;

    explicit ImageDataCudaArray(ImageFormat format, const Buffer &buffer);
    explicit ImageDataCudaArray(const NVCVImageData &data);
};

// ImageDataPitchDevice definition -----------------------
class ImageDataPitchDevice : public IImageDataPitchDevice
{
public:
    using Buffer = NVCVImageBufferPitch;

    explicit ImageDataPitchDevice(ImageFormat format, const Buffer &buffer);
    explicit ImageDataPitchDevice(const NVCVImageData &data);
};

// ImageDataPitchHost definition -----------------------
class ImageDataPitchHost : public IImageDataPitchHost
{
public:
    using Buffer = NVCVImageBufferPitch;

    explicit ImageDataPitchHost(ImageFormat format, const Buffer &buffer);
    explicit ImageDataPitchHost(const NVCVImageData &data);
};

}} // namespace nv::cv

#include "detail/ImageDataImpl.hpp"

#endif // NVCV_DETAIL_IMAGEDATA_HPP
