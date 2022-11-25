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
 * @file OpPadAndStack.hpp
 *
 * @brief Defines the private C++ class for the pad and stack operation.
 */

#ifndef NVCV_OP_PRIV_PADANDSTACK_HPP
#define NVCV_OP_PRIV_PADANDSTACK_HPP

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

class PadAndStack final : public OperatorBase
{
public:
    explicit PadAndStack();

    void operator()(cudaStream_t stream, cv::IImageBatchVarShape &in, cv::ITensor &out, cv::ITensor &top,
                    cv::ITensor &left, const NVCVBorderType borderMode, const float borderValue) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::PadAndStack> m_legacyOp;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_PADANDSTACK_HPP
