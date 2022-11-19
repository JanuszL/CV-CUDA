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
 * @file OpRotate.hpp
 *
 * @brief Defines the public C++ Class for the rotate operation.
 * @defgroup NVCV_CPP_ALGORITHM_ROTATE Rotate
 * @{
 */

#ifndef NVCV_OP_ROTATE_HPP
#define NVCV_OP_ROTATE_HPP

#include "IOperator.hpp"
#include "OpRotate.h"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/ImageFormat.hpp>
#include <nvcv/alloc/Requirements.hpp>

namespace nv { namespace cvop {

class Rotate final : public IOperator
{
public:
    explicit Rotate(const int maxVarShapeBatchSize);

    ~Rotate();

    void operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const double angleDeg, const double2 shift,
                    const NVCVInterpolationType interpolation);

    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                    cv::ITensor &angleDeg, cv::ITensor &shift, const NVCVInterpolationType interpolation);

    virtual NVCVOperatorHandle handle() const noexcept override;

private:
    NVCVOperatorHandle m_handle;
};

inline Rotate::Rotate(const int maxVarShapeBatchSize)
{
    cv::detail::CheckThrow(nvcvopRotateCreate(&m_handle, maxVarShapeBatchSize));
    assert(m_handle);
}

inline Rotate::~Rotate()
{
    nvcvOperatorDestroy(m_handle);
    m_handle = nullptr;
}

inline void Rotate::operator()(cudaStream_t stream, cv::ITensor &in, cv::ITensor &out, const double angleDeg,
                               const double2 shift, const NVCVInterpolationType interpolation)
{
    cv::detail::CheckThrow(
        nvcvopRotateSubmit(m_handle, stream, in.handle(), out.handle(), angleDeg, shift, interpolation));
}

inline void Rotate::operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                               cv::ITensor &angleDeg, cv::ITensor &shift, const NVCVInterpolationType interpolation)
{
    cv::detail::CheckThrow(nvcvopRotateVarShapeSubmit(m_handle, stream, in.handle(), out.handle(), angleDeg.handle(),
                                                      shift.handle(), interpolation));
}

inline NVCVOperatorHandle Rotate::handle() const noexcept
{
    return m_handle;
}

}} // namespace nv::cvop

#endif // NVCV_OP_ROTATE_HPP
