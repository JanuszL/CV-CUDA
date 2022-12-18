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
 * @file OpReformat.hpp
 *
 * @brief Defines the private C++ Class for the reformat operation.
 */

#ifndef NVCV_OP_PRIV_REFORMAT_HPP
#define NVCV_OP_PRIV_REFORMAT_HPP

#include "IOperator.hpp"
#include "legacy/CvCudaLegacy.h"

#include <nvcv/ITensor.hpp>

#include <memory>

namespace nvcvop::priv {

class Reformat final : public IOperator
{
public:
    explicit Reformat();

    void operator()(cudaStream_t stream, const nvcv::ITensor &in, const nvcv::ITensor &out) const;

private:
    std::unique_ptr<nvcv::legacy::cuda_op::Reformat> m_legacyOp;
};

} // namespace nvcvop::priv

#endif // NVCV_OP_PRIV_REFORMAT_HPP
