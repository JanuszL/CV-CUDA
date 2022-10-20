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

#include "TensorWrapDataPitch.hpp"

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

    int ndim = buffer.layout.ndim;

    for (int i = 0; i < ndim; ++i)
    {
        if (buffer.shape[i] < 1)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Shape #" << i << " must be >= 1, not " << buffer.shape[i];
        }
    }

    PixelType dtype{buffer.dtype};

    int firstPacked = IsChannelLast(buffer.layout) ? ndim - 2 : ndim - 1;

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

TensorWrapDataPitch::TensorWrapDataPitch(const NVCVTensorData &data, NVCVTensorDataCleanupFunc cleanup,
                                         void *ctxCleanup)
    : m_data(data)
    , m_cleanup(cleanup)
    , m_ctxCleanup(ctxCleanup)
{
    NVCV_ASSERT(data.bufferType == NVCV_TENSOR_BUFFER_PITCH_DEVICE);

    ValidateTensorBufferPitch(data.buffer.pitch);
}

TensorWrapDataPitch::~TensorWrapDataPitch()
{
    if (m_cleanup)
    {
        m_cleanup(m_ctxCleanup, &m_data);
    }
}

int32_t TensorWrapDataPitch::ndim() const
{
    return m_data.buffer.pitch.ndim;
}

const int64_t *TensorWrapDataPitch::shape() const
{
    return m_data.buffer.pitch.shape;
}

const NVCVTensorLayout &TensorWrapDataPitch::layout() const
{
    return m_data.buffer.pitch.layout;
}

DimsNCHW TensorWrapDataPitch::dims() const
{
    return ToNCHW(this->shape(), this->layout());
}

PixelType TensorWrapDataPitch::dtype() const
{
    return PixelType{m_data.buffer.pitch.dtype};
}

IAllocator &TensorWrapDataPitch::alloc() const
{
    return GetDefaultAllocator();
}

void TensorWrapDataPitch::exportData(NVCVTensorData &data) const
{
    data = m_data;
}

Version TensorWrapDataPitch::doGetVersion() const
{
    return CURRENT_VERSION;
}

} // namespace nv::cv::priv
