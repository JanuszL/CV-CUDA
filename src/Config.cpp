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

#include <nvcv/Config.h>
#include <private/core/AllocatorManager.hpp>
#include <private/core/ImageBatchManager.hpp>
#include <private/core/ImageManager.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/TensorManager.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxImageCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::ImageManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxImageBatchCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::ImageBatchManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxTensorCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::TensorManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxAllocatorCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            auto &mgr = std::get<priv::AllocatorManager &>(priv::GlobalContext().managerList());
            if (maxCount >= 0)
            {
                mgr.setFixedSize(maxCount);
            }
            else
            {
                mgr.setDynamicSize();
            }
        });
}
