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

#include <nvcv/TensorLayout.h>
#include <nvcv/TensorLayoutInfo.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/TensorLayout.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorLayoutMake, (const char *descr, NVCVTensorLayout *layout))
{
    return priv::ProtectCall(
        [&]
        {
            if (layout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to layout output must not be NULL");
            }

            *layout = priv::CreateLayout(descr);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorLayoutMakeRange,
                (const char *beg, const char *end, NVCVTensorLayout *layout))
{
    return priv::ProtectCall(
        [&]
        {
            if (layout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to layout output must not be NULL");
            }

            *layout = priv::CreateLayout(beg, end);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorLayoutMakeFirst, (NVCVTensorLayout in, int32_t n, NVCVTensorLayout *layout))
{
    return priv::ProtectCall(
        [&]
        {
            if (layout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to layout output must not be NULL");
            }

            *layout = priv::CreateFirst(in, n);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorLayoutMakeLast, (NVCVTensorLayout in, int32_t n, NVCVTensorLayout *layout))
{
    return priv::ProtectCall(
        [&]
        {
            if (layout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to layout output must not be NULL");
            }

            *layout = priv::CreateLast(in, n);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorLayoutMakeSubRange,
                (NVCVTensorLayout in, int32_t beg, int32_t end, NVCVTensorLayout *layout))
{
    return priv::ProtectCall(
        [&]
        {
            if (layout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to layout output must not be NULL");
            }

            *layout = priv::CreateSubRange(in, beg, end);
        });
}
