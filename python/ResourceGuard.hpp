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

#ifndef CVCUDA_PYTHON_RESOURCE_GUARD_HPP
#define CVCUDA_PYTHON_RESOURCE_GUARD_HPP

#include "Resource.hpp"
#include "Stream.hpp"

#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nv::cvpy {

class Resource;

class ResourceGuard
{
public:
    ResourceGuard(Stream &stream);
    ResourceGuard &add(LockMode mode, std::initializer_list<std::reference_wrapper<const Resource>> resources);

    void commit();

    ~ResourceGuard();

private:
    Stream &m_stream;

    LockResources m_resourcesPerLockMode;
};

} // namespace nv::cvpy

#endif // CVCUDA_PYTHON_RESOURCE_GUARD_HPP
