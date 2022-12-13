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
 * @file OpCvtColor.hpp
 *
 * @brief Defines the public C++ class for the CvtColor (convert color) operation.
 * @defgroup NVCV_CPP_ALGORITHM_CVTCOLOR CvtColor
 * @{
 */

#ifndef NVCV_OP_CVTCOLOR_HPP
#define NVCV_OP_CVTCOLOR_HPP

#include "IOperator.hpp"
#include "OpCvtColor.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/Size.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class CvtColor final : public IOperator
{
public:
    explicit CvtColor();

    ~CvtColor();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, NVCVColorConversionCode code);

    void operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out, NVCVColorConversionCode code);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline CvtColor::CvtColor()
{
    cv::detail::CheckThrow(nvcvopCvtColorCreate(&m_handle));
    assert(m_handle);
}

inline CvtColor::~CvtColor()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void CvtColor::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, NVCVColorConversionCode code)
{
    cv::detail::CheckThrow(nvcvopCvtColorSubmit(m_handle, stream, in.handle(), out.handle(), code));
}

inline void CvtColor::operator()(cudaStream_t stream, cv::IImageBatch &in, cv::IImageBatch &out,
                                 NVCVColorConversionCode code)
{
    cv::detail::CheckThrow(nvcvopCvtColorVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), code));
}

inline NVCVOperatorHandle CvtColor::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_CVTCOLOR_HPP
