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
