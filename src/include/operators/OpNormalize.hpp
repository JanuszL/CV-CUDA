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
 * @brief Defines the public C++ Class for the normalize operation.
 * @defgroup NVCV_CPP_ALGORITHM_NORMALIZE Normalize
 * @{
 */

#ifndef NVCV_OP_NORMALIZE_HPP
#define NVCV_OP_NORMALIZE_HPP

#include "IOperator.hpp"
#include "OpNormalize.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Normalize final : public IOperator
{
public:
    explicit Normalize();

    ~Normalize();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &base, cv::ITensor &scale, cv::ITensor &out,
                    float global_scale, float shift, float epsilon, uint32_t flags = 0);

    void operator()(cudaStream_t stream, cv::IImageBatch &in, cv::ITensor &base, cv::ITensor &scale,
                    cv::IImageBatch &out, float global_scale, float shift, float epsilon, uint32_t flags = 0);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Normalize::Normalize()
{
    cv::detail::CheckThrow(nvcvopNormalizeCreate(&m_handle));
    assert(m_handle);
}

inline Normalize::~Normalize()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Normalize::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &base, cv::ITensor &scale,
                                  cv::ITensor &out, float global_scale, float shift, float epsilon, uint32_t flags)
{
    cv::detail::CheckThrow(nvcvopNormalizeSubmit(m_handle, stream, in.handle(), base.handle(), scale.handle(),
                                                 out.handle(), global_scale, shift, epsilon, flags));
}

inline void Normalize::operator()(cudaStream_t stream, cv::IImageBatch &in, cv::ITensor &base, cv::ITensor &scale,
                                  cv::IImageBatch &out, float global_scale, float shift, float epsilon, uint32_t flags)
{
    cv::detail::CheckThrow(nvcvopNormalizeVarShapeSubmit(m_handle, stream, in.handle(), base.handle(), scale.handle(),
                                                         out.handle(), global_scale, shift, epsilon, flags));
}

inline NVCVOperatorHandle Normalize::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_NORMALIZE_HPP
