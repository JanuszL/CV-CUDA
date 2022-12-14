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

#include <nvcv/TensorLayout.h>
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
