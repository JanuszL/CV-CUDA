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

#ifndef NVCV_ITENSOR_IMPL_HPP
#define NVCV_ITENSOR_IMPL_HPP

#ifndef NVCV_ITENSOR_HPP
#    error "You must not include this header directly"
#endif

namespace nv { namespace cv {

// Implementation

inline NVCVTensorHandle ITensor::handle() const
{
    return doGetHandle();
}

inline TensorShape ITensor::shape() const
{
    NVCVTensorHandle htensor = this->handle();

    int32_t ndim = 0;
    detail::CheckThrow(nvcvTensorGetShape(htensor, &ndim, nullptr));

    NVCVTensorLayout layout;
    detail::CheckThrow(nvcvTensorGetLayout(htensor, &layout));

    TensorShape::ShapeType shape(ndim);
    detail::CheckThrow(nvcvTensorGetShape(htensor, &ndim, shape.begin()));
    return {shape, layout};
}

inline int ITensor::ndim() const
{
    int32_t ndim = 0;
    detail::CheckThrow(nvcvTensorGetShape(this->handle(), &ndim, nullptr));
    return ndim;
}

inline TensorLayout ITensor::layout() const
{
    NVCVTensorLayout layout;
    detail::CheckThrow(nvcvTensorGetLayout(this->handle(), &layout));
    return static_cast<TensorLayout>(layout);
}

inline PixelType ITensor::dtype() const
{
    NVCVPixelType out;
    detail::CheckThrow(nvcvTensorGetDataType(this->handle(), &out));
    return PixelType{out};
}

inline const ITensorData *ITensor::exportData() const
{
    NVCVTensorData data;
    detail::CheckThrow(nvcvTensorExportData(this->handle(), &data));

    if (data.bufferType != NVCV_TENSOR_BUFFER_PITCH_DEVICE)
    {
        throw Exception(Status::ERROR_INVALID_OPERATION, "Tensor data cannot be exported, buffer type not supported");
    }

    m_cacheData.emplace(TensorShape(data.shape, data.ndim, data.layout), PixelType{data.dtype}, data.buffer.pitch);

    return &*m_cacheData;
}

inline ITensor *ITensor::cast(HandleType h)
{
    return detail::CastImpl<ITensor>(&nvcvTensorGetUserPointer, &nvcvTensorSetUserPointer, h);
}

inline void ITensor::setUserPointer(void *ptr)
{
    detail::CheckThrow(nvcvTensorSetUserPointer(this->handle(), ptr));
}

inline void *ITensor::userPointer() const
{
    void *ptr;
    detail::CheckThrow(nvcvTensorGetUserPointer(this->handle(), &ptr));
    return ptr;
}

}} // namespace nv::cv

#endif // NVCV_ITENSOR_IMPL_HPP
