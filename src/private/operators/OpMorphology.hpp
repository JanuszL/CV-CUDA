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
 * @file OpMorphology.hpp
 *
 * @brief Defines the private C++ Class for the Morphology operation.
 */

#ifndef NVCV_OP_PRIV_MORPHOLOGY_HPP
#define NVCV_OP_PRIV_MORPHOLOGY_HPP

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

// Use the public nvcv API
namespace nv::cvop::priv {

class Morphology final : public OperatorBase
{
public:
    explicit Morphology();

    void operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, NVCVMorphologyType morph_type,
                    cv::Size2D mask_size, int2 anchor, int iteration, const NVCVBorderType borderMode) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::Morphology> m_legacyOp;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_MORPHOLOGY_HPP
