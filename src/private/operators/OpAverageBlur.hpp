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
 * @file OpAverageBlur.hpp
 *
 * @brief Defines the private C++ class for the AverageBlur operation.
 */

#ifndef NVCV_OP_PRIV_AVERAGEBLUR_HPP
#define NVCV_OP_PRIV_AVERAGEBLUR_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/alloc/Requirements.hpp>
#include <private/core/Exception.hpp>
#include <private/core/ICoreObject.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/Version.hpp>
#include <private/legacy/CvCudaLegacy.h>

#include <memory>

namespace nv::cvop::priv {

class AverageBlur final : public OperatorBase
{
public:
    explicit AverageBlur(cv::Size2D maxKernelSize, int maxBatchSize);

    void operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::Size2D kernelSize,
                    int2 kernelAnchor, NVCVBorderType borderMode) const;

    void operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                    const cv::ITensor &kernelSize, const cv::ITensor &kernelAnchor, NVCVBorderType borderMode) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::AverageBlur>         m_legacyOp;
    std::unique_ptr<cv::legacy::cuda_op::AverageBlurVarShape> m_legacyOpVarShape;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_AVERAGEBLUR_HPP
