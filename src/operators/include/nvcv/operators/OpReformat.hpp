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
 * @file OpReformat.hpp
 *
 * @brief Defines the public C++ Class for the reformat operation.
 * @defgroup NVCV_CPP_ALGORITHM_REFORMAT Reformat
 * @{
 */

#ifndef NVCV_OP_REFORMAT_HPP
#define NVCV_OP_REFORMAT_HPP

#include "IOperator.hpp"
#include "OpReformat.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Reformat final : public IOperator
{
public:
    explicit Reformat();

    ~Reformat();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Reformat::Reformat()
{
    cv::detail::CheckThrow(nvcvopReformatCreate(&m_handle));
    assert(m_handle);
}

inline Reformat::~Reformat()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Reformat::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out)
{
    cv::detail::CheckThrow(nvcvopReformatSubmit(m_handle, stream, in.handle(), out.handle()));
}

inline NVCVOperatorHandle Reformat::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_REFORMAT_HPP
