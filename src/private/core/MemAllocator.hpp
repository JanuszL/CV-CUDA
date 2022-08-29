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

#ifndef NVCV_PRIV_MEMALLOCATOR_HPP
#define NVCV_PRIV_MEMALLOCATOR_HPP

#include "IMemAllocator.hpp"

namespace nv::cv::priv {

class MemAllocator : public IMemAllocator
{
public:
    MemAllocator(const NVCVCustomMemAllocator *customAllocators, int32_t numCustomAllocators);

    void *allocMem(NVCVMemoryType memType, int64_t size, int32_t align) final;
    void  freeMem(NVCVMemoryType memType, void *ptr, int64_t size, int32_t align) noexcept final;

private:
    NVCVCustomMemAllocator m_allocators[NVCV_NUM_MEMORY_TYPES];

    virtual Version doGetVersion() const final;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_MEMALLOCATOR_HPP
