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

class IOperator : public nv::cv::priv::ICoreObjectHandle<IOperator, NVCVOperatorHandle>
{
};

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_IOPERATOR_HPP
