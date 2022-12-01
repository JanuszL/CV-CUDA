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
 * @file OpWarpPerspective.hpp
 *
 * @brief Defines the public C++ Class for the WarpPerspective operation.
 * @defgroup NVCV_CPP_ALGORITHM_WARP_PERSPECTIVE WarpPerspective
 * @{
 */

#ifndef NVCV_OP_WARP_PERSPECTIVE_HPP
#define NVCV_OP_WARP_PERSPECTIVE_HPP

#include "IOperator.hpp"
#include "OpWarpPerspective.h"
#include "Types.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>
#include <nvcv/cuda/math/LinAlg.hpp>

namespace cvmath = nv::cv::cuda::math;

namespace nv { namespace cvop {

class WarpPerspective final : public IOperator
{
public:
    explicit WarpPerspective();

    ~WarpPerspective();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const NVCVPerspectiveTransform transMatrix,
                    const int32_t flags, const NVCVBorderType borderMode, const float4 borderValue);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline WarpPerspective::WarpPerspective()
{
    cv::detail::CheckThrow(nvcvopWarpPerspectiveCreate(&m_handle));
    assert(m_handle);
}

inline WarpPerspective::~WarpPerspective()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void WarpPerspective::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out,
                                        const NVCVPerspectiveTransform transMatrix, const int32_t flags,
                                        const NVCVBorderType borderMode, const float4 borderValue)
{
    cv::detail::CheckThrow(nvcvopWarpPerspectiveSubmit(m_handle, stream, in.handle(), out.handle(), transMatrix, flags,
                                                       borderMode, borderValue));
}

inline NVCVOperatorHandle WarpPerspective::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_WARP_PERSPECTIVE_HPP
