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
 * @file OpWarpAffine.hpp
 *
 * @brief Defines the public C++ Class for the WarpAffine operation.
 * @defgroup NVCV_CPP_ALGORITHM_WARP_AFFINE WarpAffine
 * @{
 */

#ifndef NVCV_OP_WARP_AFFINE_HPP
#define NVCV_OP_WARP_AFFINE_HPP

#include "IOperator.hpp"
#include "OpWarpAffine.h"
#include "Types.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class WarpAffine final : public IOperator
{
public:
    explicit WarpAffine();

    ~WarpAffine();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const float trans_matrix[2 * 3],
                    const nv::cv::Size2D dsize, const int flags, const NVCVBorderType borderMode,
                    const float4 borderValue);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline WarpAffine::WarpAffine()
{
    cv::detail::CheckThrow(nvcvopWarpAffineCreate(&m_handle));
    assert(m_handle);
}

inline WarpAffine::~WarpAffine()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void WarpAffine::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out,
                                   const float trans_matrix[2 * 3], const nv::cv::Size2D dsize, const int flags,
                                   const NVCVBorderType borderMode, const float4 borderValue)
{
    cv::detail::CheckThrow(nvcvopWarpAffineSubmit(m_handle, stream, in.handle(), out.handle(), trans_matrix, dsize,
                                                  flags, borderMode, borderValue));
}

inline NVCVOperatorHandle WarpAffine::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_WARP_AFFINE_HPP
