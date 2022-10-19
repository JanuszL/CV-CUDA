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
 * @file OpCustomCrop.hpp
 *
 * @brief Defines the public C++ Class for the CustomCrop operation.
 * @defgroup NVCV_CPP_ALGORITHM_CUSTOM_CROP CustomCrop
 * @{
 */

#ifndef NVCV_OP_CUSTOM_CROP_HPP
#define NVCV_OP_CUSTOM_CROP_HPP

#include "IOperator.hpp"
#include "OpCustomCrop.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class CustomCrop final : public IOperator
{
public:
    explicit CustomCrop();

    ~CustomCrop();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const NVCVRectI cropRect);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline CustomCrop::CustomCrop()
{
    cv::detail::CheckThrow(nvcvopCustomCropCreate(&m_handle));
    assert(m_handle);
}

inline CustomCrop::~CustomCrop()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void CustomCrop::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const NVCVRectI cropRect)
{
    cv::detail::CheckThrow(nvcvopCustomCropSubmit(m_handle, stream, in.handle(), out.handle(), cropRect));
}

inline NVCVOperatorHandle CustomCrop::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_CUSTOM_CROP_HPP
