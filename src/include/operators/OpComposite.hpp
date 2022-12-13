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
 * @file OpComposite.hpp
 *
 * @brief Defines the public C++ Class for the Composite operation.
 * @defgroup NVCV_CPP_ALGORITHM_COMPOSITE Composite
 * @{
 */

#ifndef NVCV_OP_COMPOSITE_HPP
#define NVCV_OP_COMPOSITE_HPP

#include "IOperator.hpp"
#include "OpComposite.h"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Composite final : public IOperator
{
public:
    explicit Composite();

    ~Composite();

    void operator()(cudaStream_t stream, cv::ITensor &foreground, cv::ITensor &background, cv::ITensor &fgMask,
                    cv::ITensor &output);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Composite::Composite()
{
    cv::detail::CheckThrow(nvcvopCompositeCreate(&m_handle));
    assert(m_handle);
}

inline Composite::~Composite()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Composite::operator()(cudaStream_t stream, cv::ITensor &foreground, cv::ITensor &background,
                                  cv::ITensor &fgMask, cv::ITensor &output)
{
    cv::detail::CheckThrow(nvcvopCompositeSubmit(m_handle, stream, foreground.handle(), background.handle(),
                                                 fgMask.handle(), output.handle()));
}

inline NVCVOperatorHandle Composite::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_COMPOSITE_HPP
