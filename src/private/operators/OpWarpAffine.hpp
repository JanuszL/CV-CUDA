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
 * @file OpWarpAffine.hpp
 *
 * @brief Defines the private C++ Class for the reformat operation.
 */

#ifndef NVCV_OP_PRIV_WARP_AFFINE_HPP
#define NVCV_OP_PRIV_WARP_AFFINE_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <operators/OpWarpAffine.h>
#include <nvcv/alloc/Requirements.hpp>
#include <private/core/Exception.hpp>
#include <private/core/ICoreObject.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/Version.hpp>
#include <private/legacy/CvCudaLegacy.h>

#include <memory>

// Use the public nvcv API
namespace nv::cvop::priv {

class WarpAffine final : public OperatorBase
{
public:
    explicit WarpAffine();

    void operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, const float trans_matrix[2 * 3],
                    const nv::cv::Size2D dsize, const int flags, const NVCVBorderType borderMode,
                    const float4 borderValueconst) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::WarpAffine> m_legacyOp;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_WARP_AFFINE_HPP
