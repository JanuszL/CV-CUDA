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

#include <nvcv/TensorData.h>
#include <nvcv/TensorData.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/TensorData.hpp>
#include <private/core/TensorLayout.hpp>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvTensorLayoutGetNumDim, (NVCVTensorLayout layout, int32_t *ndim))
{
    return priv::ProtectCall(
        [&]
        {
            if (ndim == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "ndim output must not be NULL");
            }

            *ndim = priv::GetNumDim(layout);
        });
}
