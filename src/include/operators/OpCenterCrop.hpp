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
 * @file OpCenterCrop.hpp
 *
 * @brief Defines the public C++ Class for the CenterCrop operation.
 * @defgroup NVCV_CPP_ALGORITHM_CENTER_CROP CenterCrop
 * @{
 */

#ifndef NVCV_OP_CENTER_CROP_HPP
#define NVCV_OP_CENTER_CROP_HPP

#include "IOperator.hpp"
#include "OpCenterCrop.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class CenterCrop final : public IOperator
{
public:
    explicit CenterCrop();

    ~CenterCrop();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const cv::Size2D &cropSize);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline CenterCrop::CenterCrop()
{
    cv::detail::CheckThrow(nvcvopCenterCropCreate(&m_handle));
    assert(m_handle);
}

inline CenterCrop::~CenterCrop()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void CenterCrop::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const cv::Size2D &cropSize)
{
    cv::detail::CheckThrow(nvcvopCenterCropSubmit(m_handle, stream, in.handle(), out.handle(), cropSize.w, cropSize.h));
}

inline NVCVOperatorHandle CenterCrop::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_CENTER_CROP_HPP
