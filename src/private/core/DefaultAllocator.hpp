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

#ifndef NVCV_PRIV_CORE_DEFAULT_ALLOCATOR_HPP
#define NVCV_PRIV_CORE_DEFAULT_ALLOCATOR_HPP

#include "IAllocator.hpp"

namespace nv::cv::priv {

class DefaultAllocator final : public IAllocator
{
public:
    void *allocHostMem(int64_t size, int32_t align) override;
    void  freeHostMem(void *ptr, int64_t size, int32_t align) noexcept override;

    void *allocHostPinnedMem(int64_t size, int32_t align) override;
    void  freeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept override;

    void *allocDeviceMem(int64_t size, int32_t align) override;
    void  freeDeviceMem(void *ptr, int64_t size, int32_t align) noexcept override;

private:
    virtual Version doGetVersion() const final;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_DEFAULT_ALLOCATOR_HPP
