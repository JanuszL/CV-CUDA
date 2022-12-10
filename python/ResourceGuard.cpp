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

ResourceGuard::ResourceGuard(Stream &stream, LockMode mode,
                             std::initializer_list<std::reference_wrapper<const Resource>> resources)
    : m_stream(stream)
    , m_lockMode(mode)
{
    for (const std::reference_wrapper<const Resource> &r : resources)
    {
        r.get().submitSync(stream, mode);
        m_resources.push_back(r.get().shared_from_this());
    }
}

void ResourceGuard::commit()
{
    try
    {
        for (const std::shared_ptr<const Resource> &r : m_resources)
        {
            r->submitSignal(m_stream, m_lockMode);
        }
        m_stream.holdResources(std::move(m_resources));
    }
    catch (...)
    {
        m_stream.holdResources(std::move(m_resources));
        throw;
    }
}

ResourceGuard::~ResourceGuard()
{
    this->commit();
}

} // namespace nv::cvpy
