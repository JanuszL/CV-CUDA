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

}} // namespace nv::cv

#endif // NVCV_ITENSOR_IMPL_HPP
