/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NVCV_PRIV_CORE_DEFAULT_ALLOCATOR_HPP
#define NVCV_PRIV_CORE_DEFAULT_ALLOCATOR_HPP

#include "IAllocator.hpp"

namespace nv::cv::priv {

class DefaultAllocator final : public CoreObjectBase<IAllocator>
{
private:
    void *doAllocHostMem(int64_t size, int32_t align) override;
    void  doFreeHostMem(void *ptr, int64_t size, int32_t align) noexcept override;

    void *doAllocHostPinnedMem(int64_t size, int32_t align) override;
    void  doFreeHostPinnedMem(void *ptr, int64_t size, int32_t align) noexcept override;

    void *doAllocDeviceMem(int64_t size, int32_t align) override;
    void  doFreeDeviceMem(void *ptr, int64_t size, int32_t align) noexcept override;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_DEFAULT_ALLOCATOR_HPP
