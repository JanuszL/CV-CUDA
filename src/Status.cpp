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

#include <nvcv/Status.h>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <util/Assert.h>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvGetLastError, ())
{
    return priv::GetLastThreadError(); // noexcept
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvGetLastErrorMessage, (char *msgBuffer, int32_t lenBuffer))
{
    return priv::GetLastThreadError(msgBuffer, lenBuffer);
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPeekAtLastError, ())
{
    return priv::PeekAtLastThreadError(); // noexcept
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPeekAtLastErrorMessage, (char *msgBuffer, int32_t lenBuffer))
{
    return priv::PeekAtLastThreadError(msgBuffer, lenBuffer); // noexcept
}

NVCV_DEFINE_API(0, 0, const char *, nvcvStatusGetName, (NVCVStatus err))
{
    return priv::GetName(err); // noexcept
}

NVCV_DEFINE_API(0, 2, void, nvcvSetThreadStatusVarArgList, (NVCVStatus status, const char *fmt, va_list va))
{
    NVCVStatus ret = priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw priv::Exception(status, fmt, va);
            }
            else
            {
                throw priv::Exception(status);
            }
        });

    NVCV_ASSERT(ret == status);
}

NVCV_DEFINE_API(0, 2, void, nvcvSetThreadStatus, (NVCVStatus status, const char *fmt, ...))
{
    va_list va;
    va_start(va, fmt);

    NVCVStatus ret = priv::ProtectCall(
        [&]
        {
            if (fmt)
            {
                throw priv::Exception(status, fmt, va);
            }
            else
            {
                throw priv::Exception(status);
            }
        });

    va_end(va);

    NVCV_ASSERT(ret == status);
}
