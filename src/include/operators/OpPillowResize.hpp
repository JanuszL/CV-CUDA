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
 * @file OpPillowResize.hpp
 *
 * @brief Defines the public C++ Class for the pillow resize operation.
 * @defgroup NVCV_CPP_ALGORITHM_PILLOW_RESIZE Pillow Resize
 * @{
 */

#ifndef NVCV_OP_PILLOW_RESIZE_HPP
#define NVCV_OP_PILLOW_RESIZE_HPP

#include "IOperator.hpp"
#include "OpPillowResize.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class PillowResize final : public IOperator
{
public:
    explicit PillowResize(cv::Size2D maxSize, int maxBatchSize, cv::ImageFormat fmt);

    ~PillowResize();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const NVCVInterpolationType interpolation);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline PillowResize::PillowResize(cv::Size2D maxSize, int maxBatchSize, cv::ImageFormat fmt)
{
    NVCVImageFormat cfmt = fmt.cvalue();
    cv::detail::CheckThrow(nvcvopPillowResizeCreate(&m_handle, maxSize.w, maxSize.h, maxBatchSize, cfmt));
    assert(m_handle);
}

inline PillowResize::~PillowResize()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void PillowResize::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out,
                                     const NVCVInterpolationType interpolation)
{
    cv::detail::CheckThrow(nvcvopPillowResizeSubmit(m_handle, stream, in.handle(), out.handle(), interpolation));
}

inline NVCVOperatorHandle PillowResize::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_PILLOW_RESIZE_HPP
