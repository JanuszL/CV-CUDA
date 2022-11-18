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

#include "StreamStack.hpp"

#include "Stream.hpp"

namespace nv::cvpy {

void StreamStack::push(Stream &stream)
{
    std::unique_lock lk(m_mtx);
    m_stack.push(stream.shared_from_this());
}

void StreamStack::pop()
{
    std::unique_lock lk(m_mtx);
    m_stack.pop();
}

std::shared_ptr<Stream> StreamStack::top()
{
    std::unique_lock lk(m_mtx);
    if (!m_stack.empty())
    {
        return m_stack.top().lock();
    }
    else
    {
        return nullptr;
    }
}

StreamStack &StreamStack::Instance()
{
    static StreamStack stack;
    return stack;
}

} // namespace nv::cvpy
