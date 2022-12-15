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

#include "TensorWrapDataStrided.hpp"

#include "IAllocator.hpp"
#include "Requirements.hpp"
#include "TensorData.hpp"
#include "TensorLayout.hpp"

#include <cuda_runtime.h>
#include <fmt/DataType.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nv::cv::priv {

static void ValidateTensorBufferStrided(const NVCVTensorData &tdata)
{
    NVCV_ASSERT(tdata.bufferType == NVCV_TENSOR_BUFFER_STRIDED_DEVICE);

    const NVCVTensorBufferStrided &buffer = tdata.buffer.strided;

    if (buffer.basePtr == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Memory buffer must not be NULL";
    }

    int ndim = tdata.ndim;

    if (ndim <= 0)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Number of dimensions must be >= 1, not %d", ndim);
    }

    for (int i = 0; i < ndim; ++i)
    {
        if (tdata.shape[i] < 1)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Shape #" << i << " must be >= 1, not " << tdata.shape[i];
        }
    }

    DataType dtype{tdata.dtype};

    int firstPacked = IsChannelLast(tdata.layout) ? std::max(0, ndim - 2) : ndim - 1;

    // Test packed dimensions
    int dim;
    for (dim = ndim - 1; dim >= firstPacked; --dim)
    {
        int correctPitch = dim == ndim - 1 ? dtype.strideBytes() : buffer.strides[dim + 1] * tdata.shape[dim + 1];
        if (buffer.strides[dim] != correctPitch)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Pitch of dimension " << dim << " must be == " << correctPitch << " (packed)"
                << ", but it is " << buffer.strides[dim];
        }
    }

    // Test non-packed dimensions
    for (; dim >= 0; --dim)
    {
        int minPitch = buffer.strides[dim + 1] * tdata.shape[dim + 1];
        if (buffer.strides[dim] < minPitch)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Pitch of dimension " << dim << " must be >= " << minPitch << ", but it is " << buffer.strides[dim];
        }
    }
}

TensorWrapDataStrided::TensorWrapDataStrided(const NVCVTensorData &tdata, NVCVTensorDataCleanupFunc cleanup,
                                             void *ctxCleanup)
    : m_tdata(tdata)
    , m_cleanup(cleanup)
    , m_ctxCleanup(ctxCleanup)
{
    ValidateTensorBufferStrided(tdata);
}

TensorWrapDataStrided::~TensorWrapDataStrided()
{
    if (m_cleanup)
    {
        m_cleanup(m_ctxCleanup, &m_tdata);
    }
}

int32_t TensorWrapDataStrided::ndim() const
{
    return m_tdata.ndim;
}

const int64_t *TensorWrapDataStrided::shape() const
{
    return m_tdata.shape;
}

const NVCVTensorLayout &TensorWrapDataStrided::layout() const
{
    return m_tdata.layout;
}

DataType TensorWrapDataStrided::dtype() const
{
    return DataType{m_tdata.dtype};
}

IAllocator &TensorWrapDataStrided::alloc() const
{
    return GetDefaultAllocator();
}

void TensorWrapDataStrided::exportData(NVCVTensorData &tdata) const
{
    tdata = m_tdata;
}

} // namespace nv::cv::priv
