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
    void *allocHostMem(int64_t size, int32_t align);
    void  freeHostMem(void *ptr, int64_t size, int32_t align) noexcept;

    void *allocHostPinnedMem(int64_t size, int32_t align);
    void  freeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept;

    void *allocDeviceMem(int64_t size, int32_t align);
    void  freeDeviceMem(void *ptr, int64_t size, int32_t align) noexcept;

private:
    // NVI idiom
    virtual void *doAllocHostMem(int64_t size, int32_t align)                    = 0;
    virtual void  doFreeHostMem(void *ptr, int64_t size, int32_t align) noexcept = 0;

    virtual void *doAllocHostPinnedMem(int64_t size, int32_t align)                    = 0;
    virtual void  doFreeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept = 0;

    virtual void *doAllocDeviceMem(int64_t size, int32_t align)                    = 0;
    virtual void  doFreeDeviceMem(void *ptr, int64_t size, int32_t align) noexcept = 0;
};

priv::IAllocator &GetAllocator(NVCVAllocator handle);
priv::IAllocator &GetDefaultAllocator();

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_IALLOCATOR_HPP
