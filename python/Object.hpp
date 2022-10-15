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

#ifndef NVCV_PYTHON_OBJECT_HPP
#define NVCV_PYTHON_OBJECT_HPP

#include <memory>

namespace nv::cvpy {

// Parent of all VPI objects that are reference-counted
class Object : public std::enable_shared_from_this<Object>
{
public:
    virtual ~Object() = 0;

    Object(Object &&) = delete;

protected:
    Object() = default;
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_OBJECT_HPP
