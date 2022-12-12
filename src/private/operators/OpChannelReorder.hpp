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
 * @file OpChannelReorder.hpp
 *
 * @brief Defines the private C++ Class for the channel reorder operation.
 */

#ifndef NVCV_OP_PRIV_CHANNEL_REORDER_HPP
#define NVCV_OP_PRIV_CHANNEL_REORDER_HPP

#include "IOperator.hpp"

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <private/legacy/CvCudaLegacy.h>

#include <memory>

namespace nv::cvop::priv {

class ChannelReorder final : public OperatorBase
{
public:
    explicit ChannelReorder();

    void operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, cv::IImageBatchVarShape &out,
                    const cv::ITensor &orders) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::ChannelReorderVarShape> m_legacyOpVarShape;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_CHANNEL_REORDER_HPP
