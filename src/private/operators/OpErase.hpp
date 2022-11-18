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
 * @file OpErase.hpp
 *
 * @brief Defines the private C++ Class for the erase operation.
 */

#ifndef NVCV_OP_PRIV_ERASE_HPP
#define NVCV_OP_PRIV_ERASE_HPP

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

class Erase final : public IOperator
{
public:
    explicit Erase();

    void operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, cv::ITensor &anchor_x,
                    cv::ITensor &anchor_y, cv::ITensor &erasing_w, cv::ITensor &erasing_h, cv::ITensor &erasing_c,
                    cv::ITensor &values, cv::ITensor &imgIdx, int max_eh, int max_ew, bool random, unsigned int seed,
                    bool inplace) const;

    cv::priv::Version doGetVersion() const override;

private:
    std::unique_ptr<cv::legacy::cuda_op::Erase> m_legacyOp;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_ERASE_HPP
