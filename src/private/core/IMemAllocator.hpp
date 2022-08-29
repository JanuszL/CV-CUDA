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

#ifndef NVCV_PRIV_IMEMALLOCATOR_HPP
#define NVCV_PRIV_IMEMALLOCATOR_HPP

#include "ICoreObject.hpp"

#include <cuda_runtime.h>
#include <nvcv/MemAllocator.h>

#include <cstddef> // for std::max_align_t
#include <functional>

namespace nv::cv::priv {

class IMemAllocator : public ICoreObjectHandle<IMemAllocator, NVCVMemAllocator>
{
public:
    virtual void *allocMem(NVCVMemoryType memType, int64_t size, int32_t align) = 0;

    virtual void freeMem(NVCVMemoryType memType, void *ptr, int64_t size, int32_t align) noexcept = 0;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_IMEMALLOCATOR_HPP
