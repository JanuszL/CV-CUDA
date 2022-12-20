/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "Version.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/operators/Operator.h>

namespace nv::cvop::priv {

class IOperator
{
public:
    using HandleType    = NVCVOperatorHandle;
    using InterfaceType = IOperator;

    virtual ~IOperator() = default;

    HandleType handle() const
    {
        return reinterpret_cast<HandleType>(const_cast<IOperator *>(static_cast<const IOperator *>(this)));
    }

    Version version()
    {
        return CURRENT_VERSION;
    }
};

IOperator *ToOperatorPtr(void *handle);

template<class T>
inline T *ToDynamicPtr(NVCVOperatorHandle h)
{
    return dynamic_cast<T *>(ToOperatorPtr(h));
}

template<class T>
inline T &ToDynamicRef(NVCVOperatorHandle h)
{
    if (h == nullptr)
    {
        throw cv::Exception(cv::Status::ERROR_INVALID_ARGUMENT, "Handle cannot be NULL");
    }

    if (T *child = ToDynamicPtr<T>(h))
    {
        return *child;
    }
    else
    {
        throw cv::Exception(cv::Status::ERROR_NOT_COMPATIBLE,
                            "Handle doesn't correspond to the requested object or was already destroyed.");
    }
}

} // namespace nv::cvop::priv

#endif // NVCV_OP_PRIV_IOPERATOR_HPP
