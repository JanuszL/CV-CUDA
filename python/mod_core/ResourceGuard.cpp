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

#include "ResourceGuard.hpp"

#include "Resource.hpp"

namespace nv::cvpy {

ResourceGuard::ResourceGuard(Stream &stream)
    : m_stream(stream)
{
}

ResourceGuard &ResourceGuard::add(LockMode                                                      mode,
                                  std::initializer_list<std::reference_wrapper<const Resource>> resources)
{
    for (const std::reference_wrapper<const Resource> &r : resources)
    {
        r.get().submitSync(m_stream, mode);
        m_resourcesPerLockMode.emplace(mode, r.get().shared_from_this());
    }
    return *this;
}

void ResourceGuard::commit()
{
    try
    {
        for (auto &[lockMode, res] : m_resourcesPerLockMode)
        {
            res->submitSignal(m_stream, lockMode);
        }
        m_stream.holdResources(std::move(m_resourcesPerLockMode));
    }
    catch (...)
    {
        m_stream.holdResources(std::move(m_resourcesPerLockMode));
        throw;
    }
}

ResourceGuard::~ResourceGuard()
{
    this->commit();
}

} // namespace nv::cvpy
