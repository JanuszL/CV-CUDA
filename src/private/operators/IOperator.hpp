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
 * @file IOperator.hpp
 *
 * @brief Defines the private C++ Class for the operator interface.
 */

#ifndef NVCV_OP_PRIV_IOPERATOR_HPP
#define NVCV_OP_PRIV_IOPERATOR_HPP

#include <nvcv/IImage.hpp>
#include <operators/IOperator.hpp>
#include <private/core/ICoreObject.hpp>

namespace nv::cvop::priv {

class IOperator : public cv::priv::ICoreObjectHandle<IOperator, NVCVOperatorHandle>
{
};

class OperatorBase : public cv::priv::CoreObjectBase<IOperator>
{
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_IOPERATOR_HPP
