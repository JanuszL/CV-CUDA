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
            if (maxCount >= 0)
            {
                priv::ImageManager::Instance().setFixedSize(maxCount);
            }
            else
            {
                priv::ImageManager::Instance().setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxImageBatchCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            if (maxCount >= 0)
            {
                priv::ImageBatchManager::Instance().setFixedSize(maxCount);
            }
            else
            {
                priv::ImageBatchManager::Instance().setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxTensorCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            if (maxCount >= 0)
            {
                priv::TensorManager::Instance().setFixedSize(maxCount);
            }
            else
            {
                priv::TensorManager::Instance().setDynamicSize();
            }
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvConfigSetMaxAllocatorCount, (int32_t maxCount))
{
    return priv::ProtectCall(
        [&]
        {
            if (maxCount >= 0)
            {
                priv::AllocatorManager::Instance().setFixedSize(maxCount);
            }
            else
            {
                priv::AllocatorManager::Instance().setDynamicSize();
            }
        });
}
