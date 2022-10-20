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

#include "TensorWrapData.hpp"

#include "IAllocator.hpp"
#include "Requirements.hpp"
#include "TensorData.hpp"
#include "TensorLayout.hpp"

#include <cuda_runtime.h>
#include <fmt/PixelType.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nv::cv::priv {

static void ValidateTensorBufferPitch(const NVCVTensorBufferPitch &buffer)
{
    if (buffer.mem == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Memory buffer must not be NULL";
    }

    int ndim = GetNumDim(buffer.layout);

    for (int i = 0; i < ndim; ++i)
    {
        if (buffer.shape[i] < 1)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Shape #" << i << " must be >= 1, not " << buffer.shape[i];
        }
    }

    PixelType dtype{buffer.dtype};

    int firstPacked;
    switch (buffer.layout)
    {
    case NVCV_TENSOR_NCHW:
        firstPacked = ndim - 1;
        break;
    case NVCV_TENSOR_NHWC:
        firstPacked = ndim - 2;
        break;
    default:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid tensor layout: " << buffer.layout;
    }

    // Test packed dimensions
    int dim;
    for (dim = ndim - 1; dim >= firstPacked; --dim)
    {
        int correctPitch = dim == ndim - 1 ? dtype.strideBytes() : buffer.pitchBytes[dim + 1] * buffer.shape[dim + 1];
        if (buffer.pitchBytes[dim] != correctPitch)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Pitch of dimension " << dim << " must be == " << correctPitch << " (packed)"
                << ", but it is " << buffer.pitchBytes[dim];
        }
    }

    // Test non-packed dimensions
    for (; dim >= 0; --dim)
    {
        int minPitch = buffer.pitchBytes[dim + 1] * buffer.shape[dim + 1];
        if (buffer.pitchBytes[dim] < minPitch)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Pitch of dimension " << dim << " must be >= " << minPitch
                                                         << ", but it is " << buffer.pitchBytes[dim];
        }
    }
}

TensorWrapData::TensorWrapData(const NVCVTensorData &data, NVCVTensorDataCleanupFunc cleanup, void *ctxCleanup)
    : m_data(data)
    , m_cleanup(cleanup)
    , m_ctxCleanup(ctxCleanup)
{
    switch (data.bufferType)
    {
    case NVCV_TENSOR_BUFFER_PITCH_DEVICE:
        ValidateTensorBufferPitch(data.buffer.pitch);
        return;

    case NVCV_TENSOR_BUFFER_NONE:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid wrapping of buffer type NONE";
    }

    throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image buffer type not supported";
}

TensorWrapData::~TensorWrapData()
{
    if (m_cleanup)
    {
        m_cleanup(m_ctxCleanup, &m_data);
    }
}

static const Shape g_emptyShape = {};

const Shape &TensorWrapData::shape() const
{
    switch (m_data.bufferType)
    {
    case NVCV_TENSOR_BUFFER_PITCH_DEVICE:
        // TODO: This is UB under strict C++ rules
        return *reinterpret_cast<const Shape *>(m_data.buffer.pitch.shape);

    case NVCV_TENSOR_BUFFER_NONE:
        return g_emptyShape;
    }

    NVCV_ASSERT(!"Invalid buffer type");
    return g_emptyShape;
}

NVCVTensorLayout TensorWrapData::layout() const
{
    switch (m_data.bufferType)
    {
    case NVCV_TENSOR_BUFFER_PITCH_DEVICE:
        return m_data.buffer.pitch.layout;

    case NVCV_TENSOR_BUFFER_NONE:
        return NVCV_TENSOR_NCHW;
    }

    NVCV_ASSERT(!"Invalid buffer type");
    return NVCV_TENSOR_NCHW;
}

DimsNCHW TensorWrapData::dims() const
{
    return ToNCHW(this->shape(), this->layout());
}

PixelType TensorWrapData::dtype() const
{
    switch (m_data.bufferType)
    {
    case NVCV_TENSOR_BUFFER_PITCH_DEVICE:
        return PixelType{m_data.buffer.pitch.dtype};

    case NVCV_TENSOR_BUFFER_NONE:
        return PixelType{NVCV_PIXEL_TYPE_NONE};
    }

    NVCV_ASSERT(!"Invalid buffer type");
    return PixelType{NVCV_PIXEL_TYPE_NONE};
}

IAllocator &TensorWrapData::alloc() const
{
    return GetDefaultAllocator();
}

void TensorWrapData::exportData(NVCVTensorData &data) const
{
    data = m_data;
}

Version TensorWrapData::doGetVersion() const
{
    return CURRENT_VERSION;
}

} // namespace nv::cv::priv
