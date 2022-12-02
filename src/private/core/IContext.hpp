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

#ifndef NVCV_PRIV_CORE_ICONTEXT_HPP
#define NVCV_PRIV_CORE_ICONTEXT_HPP

#include <nvcv/Fwd.h>

#include <tuple>

namespace nv::cv::priv {

// Forward declaration
template<class HandleType>
class CoreObjManager;

using ImageManager      = CoreObjManager<NVCVImageHandle>;
using ImageBatchManager = CoreObjManager<NVCVImageBatchHandle>;
using TensorManager     = CoreObjManager<NVCVTensorHandle>;
using AllocatorManager  = CoreObjManager<NVCVAllocatorHandle>;

class IAllocator;

class IContext
{
public:
    using Managers = std::tuple<AllocatorManager &, ImageManager &, ImageBatchManager &, TensorManager &>;

    template<class HandleType>
    CoreObjManager<HandleType> &manager()
    {
        return std::get<CoreObjManager<HandleType> &>(managerList());
    }

    virtual const Managers &managerList() const = 0;
    virtual IAllocator     &allocDefault()      = 0;
};

// Defined in Context.cpp
IContext &GlobalContext();

} // namespace nv::cv::priv

#endif // NVCV_PRIV_CORE_ICONTEXT_HPP
