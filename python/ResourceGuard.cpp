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
