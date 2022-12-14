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

/**
 * @file OpErase.hpp
 *
 * @brief Defines the public C++ Class for the erase operation.
 * @defgroup NVCV_CPP_ALGORITHM_ERASE Erase
 * @{
 */

#ifndef NVCV_OP_ERASE_HPP
#define NVCV_OP_ERASE_HPP

#include "IOperator.hpp"
#include "OpErase.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Erase final : public IOperator
{
public:
    explicit Erase(int32_t max_num_erasing_area);

    ~Erase();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::ITensor &anchor, cv::ITensor &erasing,
                    cv::ITensor &values, cv::ITensor &imgIdx, bool random, uint32_t seed);

    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out, cv::ITensor &anchor,
                    cv::ITensor &erasing, cv::ITensor &values, cv::ITensor &imgIdx, bool random, uint32_t seed);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Erase::Erase(int32_t max_num_erasing_area)
{
    cv::detail::CheckThrow(nvcvopEraseCreate(&m_handle, max_num_erasing_area));
    assert(m_handle);
}

inline Erase::~Erase()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Erase::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, cv::ITensor &anchor,
                              cv::ITensor &erasing, cv::ITensor &values, cv::ITensor &imgIdx, bool random,
                              uint32_t seed)
{
    cv::detail::CheckThrow(nvcvopEraseSubmit(m_handle, stream, in.handle(), out.handle(), anchor.handle(),
                                             erasing.handle(), values.handle(), imgIdx.handle(), random, seed));
}

inline void Erase::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                              cv::ITensor &anchor, cv::ITensor &erasing, cv::ITensor &values, cv::ITensor &imgIdx,
                              bool random, uint32_t seed)
{
    cv::detail::CheckThrow(nvcvopEraseVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), anchor.handle(),
                                                     erasing.handle(), values.handle(), imgIdx.handle(), random, seed));
}

inline NVCVOperatorHandle Erase::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_ERASE_HPP
