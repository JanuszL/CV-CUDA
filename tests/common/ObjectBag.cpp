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

#include "ObjectBag.hpp"

#include <nvcv/MemAllocator.h>

namespace nv::cv::test {

ObjectBag::~ObjectBag()
{
    // Destroy from back to front
    while (!m_objs.empty())
    {
        m_objs.top()(); // call object destructor
        m_objs.pop();
    }
}

void ObjectBag::insert(NVCVMemAllocator handle)
{
    m_objs.push([handle]() { nvcvMemAllocatorDestroy(handle); });
}

} // namespace nv::cv::test
