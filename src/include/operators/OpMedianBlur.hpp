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
 * @file OpMedianBlur.hpp
 *
 * @brief Defines the public C++ Class for the median blur operation.
 * @defgroup NVCV_CPP_ALGORITHM_MEDIAN_BLUR MedianBlur
 * @{
 */

#ifndef NVCV_OP_MEDIAN_BLUR_HPP
#define NVCV_OP_MEDIAN_BLUR_HPP

#include "IOperator.hpp"
#include "OpMedianBlur.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class MedianBlur final : public IOperator
{
public:
    explicit MedianBlur(const int maxVarShapeBatchSize);

    ~MedianBlur();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const cv::Size2D ksize);

    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out, cv::ITensor &ksize);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline MedianBlur::MedianBlur(const int maxVarShapeBatchSize)
{
    cv::detail::CheckThrow(nvcvopMedianBlurCreate(&m_handle, maxVarShapeBatchSize));
    assert(m_handle);
}

inline MedianBlur::~MedianBlur()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void MedianBlur::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const cv::Size2D ksize)
{
    cv::detail::CheckThrow(nvcvopMedianBlurSubmit(m_handle, stream, in.handle(), out.handle(), ksize.w, ksize.h));
}

inline void MedianBlur::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                                   cv::ITensor &ksize)
{
    cv::detail::CheckThrow(nvcvopMedianBlurVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), ksize.handle()));
}

inline NVCVOperatorHandle MedianBlur::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_MEDIAN_BLUR_HPP
