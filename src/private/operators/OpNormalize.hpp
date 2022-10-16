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
 * @file OpNormalize.hpp
 *
 * @brief Defines the private C++ Class for the reformat operation.
 */

#ifndef NVCV_OP_PRIV_NORMALIZE_HPP
#define NVCV_OP_PRIV_NORMALIZE_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/ITensor.hpp>
#include <nvcv/alloc/Requirements.hpp>
#include <private/core/Exception.hpp>
#include <private/core/ICoreObject.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/Version.hpp>
#include <private/legacy/CvCudaLegacy.h>

#include <memory>

// Use the public nvcv API
namespace nv::cv::op::priv {

class Normalize final : public IOperator
{
public:
    explicit Normalize();

    void operator()(cudaStream_t stream, const ITensor &in, const ITensor &out,
                    bool scale_is_stddev, float global_scale, float shift, float epsilon) const;

    cv::priv::Version doGetVersion() const override;

private:
    //std::unique_ptr<legacy::cuda_op::Normalize> m_legacyOp;
};

} // namespace nv::cv::op::priv

#endif // NVCV_OP_PRIV_NORMALIZE_HPP
