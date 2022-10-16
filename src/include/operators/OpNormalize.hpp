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

/**
 * @file OpNormalize.hpp
 *
 * @brief Defines the public C++ Class for the reformat operation.
 * @defgroup NVCV_CPP_ALGORITHM_NORMALIZE Normalize
 * @{
 */

#ifndef NVCV_OP_NORMALIZE_HPP
#define NVCV_OP_NORMALIZE_HPP

#include "IOperator.hpp"
#include "OpNormalize.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

using namespace nv::cv;

namespace nv { namespace cv_op {

class Normalize final : public IOperator
{
public:
    explicit Normalize();

    ~Normalize();

    void operator()(cudaStream_t stream, ITensor &in, ITensor &out, bool scale_is_stddev, float global_scale,
                     float shift, float epsilon);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Normalize::Normalize()
{
    detail::CheckThrow(nvcvopNormalizeCreate(&m_handle));
    assert(m_handle);
}

inline Normalize::~Normalize()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Normalize::operator()(cudaStream_t stream, ITensor &in, ITensor &out,  bool scale_is_stddev, float global_scale,
                     float shift, float epsilon)
{
    detail::CheckThrow(nvcvopNormalizeSubmit(m_handle, stream, in.handle(), out.handle(),  scale_is_stddev, global_scale,
                      shift,  epsilon));
}

inline NVCVOperatorHandle Normalize::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cv_op

#endif // NVCV_OP_NORMALIZE_HPP
