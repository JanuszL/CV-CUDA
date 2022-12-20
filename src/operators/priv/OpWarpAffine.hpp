/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file OpWarpAffine.hpp
 *
 * @brief Defines the private C++ Class for the warp affine operation.
 */

#ifndef NVCV_OP_PRIV_WARP_AFFINE_HPP
#define NVCV_OP_PRIV_WARP_AFFINE_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/IImageBatch.hpp>
#include <nvcv/ITensor.hpp>
#include <nvcv/operators/OpWarpAffine.h>

#include <memory>

namespace nv::cvop::priv {

class WarpAffine final : public IOperator
{
public:
    explicit WarpAffine(const int32_t maxVarShapeBatchSize);

    void operator()(cudaStream_t stream, const cv::ITensor &in, const cv::ITensor &out, const NVCVAffineTransform xform,
                    const int32_t flags, const NVCVBorderType borderMode, const float4 borderValueconst) const;

    void operator()(cudaStream_t stream, const cv::IImageBatchVarShape &in, const cv::IImageBatchVarShape &out,
                    const cv::ITensor &transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                    const float4 borderValue) const;

private:
    std::unique_ptr<cv::legacy::cuda_op::WarpAffine>         m_legacyOp;
    std::unique_ptr<cv::legacy::cuda_op::WarpAffineVarShape> m_legacyOpVarShape;
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_WARP_AFFINE_HPP
