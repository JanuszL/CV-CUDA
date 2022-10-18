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
 * @brief Defines the public C++ interface to operator interfaces.
 */

#ifndef NVCV_OP_IOPERATOR_HPP
#define NVCV_OP_IOPERATOR_HPP

#include "Operator.h"

namespace nv { namespace cvop {

class IOperator
{
public:
    virtual ~IOperator() = default;

    virtual NVCVOperatorHandle handle() const noexcept = 0;

private:
};

}} // namespace nv::cvop

#endif // NVCV_OP_IOPERATOR_HPP
