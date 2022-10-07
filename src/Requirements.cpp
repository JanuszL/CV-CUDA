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

#include <nvcv/alloc/Requirements.h>
#include <private/core/Exception.hpp>
#include <private/core/Requirements.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvRequirementsInit, (NVCVRequirements * reqs))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqs == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to requirements must not be NULL");
            }

            priv::Init(*reqs);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvRequirementsAdd, (NVCVRequirements * reqSum, const NVCVRequirements *req))
{
    return priv::ProtectCall(
        [&]
        {
            if (reqSum == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to summed requirements must not be NULL");
            }

            if (req == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input requirements must not be NULL");
            }

            priv::Add(*reqSum, *req);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemRequirementsCalcTotalSizeBytes,
                (const NVCVMemRequirements *memReq, int64_t *sizeBytes))
{
    return priv::ProtectCall(
        [&]
        {
            if (memReq == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to the memory requirements must not be NULL");
            }

            if (sizeBytes == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to the size output must not be NULL");
            }

            *sizeBytes = priv::CalcTotalSizeBytes(*memReq);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMemRequirementsAddBuffer,
                (NVCVMemRequirements * memReq, int64_t bufSize, int64_t bufAlignment))
{
    return priv::ProtectCall(
        [&]
        {
            if (memReq == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to the memory requirements must not be NULL");
            }

            priv::AddBuffer(*memReq, bufSize, bufAlignment);
        });
}
