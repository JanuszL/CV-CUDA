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

#ifndef NVVPI_TEST_UTIL_OBJECTBAG_HPP
#define NVVPI_TEST_UTIL_OBJECTBAG_HPP

#include <nvcv/alloc/Fwd.h>

#include <functional>
#include <stack>

namespace nv::cv::test {

// Bag of NVCV objects, destroys them in its dtor in reverse
// order of insertion.
class ObjectBag final
{
public:
    ObjectBag()                  = default;
    ObjectBag(const ObjectBag &) = delete;

    ~ObjectBag();

    void insert(NVCVAllocator *handle);

private:
    std::stack<std::function<void()>> m_objs;
};

} // namespace nv::cv::test

#endif // NVVPI_TEST_UTIL_OBJECTBAG_HPP
