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

#ifndef NVCV_PYTHON_STREAMSTACK_HPP
#define NVCV_PYTHON_STREAMSTACK_HPP

#include <memory>
#include <mutex>
#include <stack>

namespace nv::cvpy {

class Stream;

class StreamStack
{
public:
    void                    push(Stream &stream);
    void                    pop();
    std::shared_ptr<Stream> top();

    static StreamStack &Instance();

private:
    std::stack<std::weak_ptr<Stream>> m_stack;
    std::weak_ptr<Stream>             m_cur;
    std::mutex                        m_mtx;
};

} // namespace nv::cvpy

#endif // NVCV_PYTHON_STREAMSTACK_HPP
