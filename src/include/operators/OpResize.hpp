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
 * @file OpResize.hpp
 *
 * @brief Defines the public C++ Class for the resize operation.
 * @defgroup NVCV_CPP_ALGORITHM_RESIZE Resize
 * @{
 */

#ifndef NVCV_OP_RESIZE_HPP
#define NVCV_OP_RESIZE_HPP

#include "IOperator.hpp"
#include "OpResize.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Resize final : public IOperator
{
public:
    explicit Resize();

    ~Resize();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const NVCVInterpolationType interpolation);
    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                    const NVCVInterpolationType interpolation);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Resize::Resize()
{
    cv::detail::CheckThrow(nvcvopResizeCreate(&m_handle));
    assert(m_handle);
}

inline Resize::~Resize()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Resize::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out,
                               const NVCVInterpolationType interpolation)
{
    cv::detail::CheckThrow(nvcvopResizeSubmit(m_handle, stream, in.handle(), out.handle(), interpolation));
}

inline void Resize::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                               const NVCVInterpolationType interpolation)
{
    cv::detail::CheckThrow(nvcvopResizeVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), interpolation));
}

inline NVCVOperatorHandle Resize::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_RESIZE_HPP
