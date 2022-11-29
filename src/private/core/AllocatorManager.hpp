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

#ifndef NVCV_PRIV_CORE_ALLOCATORMANAGER_HPP
#define NVCV_PRIV_CORE_ALLOCATORMANAGER_HPP

#include "CustomAllocator.hpp"
#include "DefaultAllocator.hpp"
#include "IContext.hpp"

namespace nv::cv::priv {

using AllocatorManager = CoreObjManager<NVCVAllocatorHandle>;

using AllocatorStorage = CompatibleStorage<DefaultAllocator, CustomAllocator>;

template<>
class CoreObjManager<NVCVAllocatorHandle> : public HandleManager<IAllocator, AllocatorStorage>
{
    using Base = HandleManager<IAllocator, AllocatorStorage>;

public:
    using Base::Base;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_ALLOCATORMANAGER_HPP
