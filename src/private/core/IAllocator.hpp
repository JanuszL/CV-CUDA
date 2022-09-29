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

#ifndef NVCV_PRIV_CORE_IALLOCATOR_HPP
#define NVCV_PRIV_CORE_IALLOCATOR_HPP

#include "ICoreObject.hpp"

#include <nvcv/alloc/Fwd.h>

namespace nv::cv::priv {

class IAllocator : public ICoreObjectHandle<IAllocator, NVCVAllocator>
{
public:
    virtual void *allocHostMem(int64_t size, int32_t align)                    = 0;
    virtual void  freeHostMem(void *ptr, int64_t size, int32_t align) noexcept = 0;

    virtual void *allocHostPinnedMem(int64_t size, int32_t align)                    = 0;
    virtual void  freeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept = 0;

    virtual void *allocDeviceMem(int64_t size, int32_t align)                    = 0;
    virtual void  freeDeviceMem(void *ptr, int64_t size, int32_t align) noexcept = 0;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_IALLOCATOR_HPP
