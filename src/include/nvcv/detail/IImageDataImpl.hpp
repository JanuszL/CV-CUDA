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

#ifndef NVCV_IIMAGEDATA_IMPL_HPP
#define NVCV_IIMAGEDATA_IMPL_HPP

#ifndef NVCV_IIMAGEDATA_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// Implementation - IImageData ---------------------------------
inline IImageData::IImageData(const NVCVImageData &data)
    : m_data(data)
{
}

inline IImageData::~IImageData()
{
    // required dtor implementation
}

inline const NVCVImageData &IImageData::cdata() const
{
    return m_data;
}

inline NVCVImageData &IImageData::cdata()
{
    return m_data;
}

inline ImageFormat IImageData::format() const
{
    return ImageFormat{this->cdata().format};
}

// Implementation - IImageDataCudaArray -------------------------

inline IImageDataCudaArray::~IImageDataCudaArray()
{
    // required dtor implementation
}

inline int32_t IImageDataCudaArray::numPlanes() const
{
    const NVCVImageBufferCudaArray &data = this->cdata().buffer.cudaarray;
    return data.numPlanes;
}

inline cudaArray_t IImageDataCudaArray::plane(int p) const
{
    const NVCVImageBufferCudaArray &data = this->cdata().buffer.cudaarray;

    if (p < 0 || p >= data.numPlanes)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Plane out of bounds");
    }
    return data.planes[p];
}

// Implementation - IImageDataPitch ------------------------------

inline IImageDataPitch::~IImageDataPitch()
{
    // required dtor implementation
}

inline Size2D IImageDataPitch::size() const
{
    const NVCVImageBufferPitch &data = this->cdata().buffer.pitch;
    if (data.numPlanes > 0)
    {
        return {data.planes[0].width, data.planes[0].height};
    }
    else
    {
        return {0, 0};
    }
}

inline int32_t IImageDataPitch::numPlanes() const
{
    const NVCVImageBufferPitch &data = this->cdata().buffer.pitch;
    return data.numPlanes;
}

inline const ImagePlanePitch &IImageDataPitch::plane(int p) const
{
    const NVCVImageBufferPitch &data = this->cdata().buffer.pitch;
    if (p < 0 || p >= data.numPlanes)
    {
        throw Exception(Status::ERROR_INVALID_ARGUMENT, "Plane out of bounds");
    }
    return data.planes[p];
}

// Implementation - IImageDataPitchDevice ------------------------------
inline IImageDataPitchDevice::~IImageDataPitchDevice()
{
    // required dtor implementation
}

// Implementation - IImageDataPitchHost ------------------------------
inline IImageDataPitchHost::~IImageDataPitchHost()
{
    // required dtor implementation
}

}} // namespace nv::cv

#endif // NVCV_IIMAGEDATA_IMPL_HPP