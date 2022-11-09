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

#include "Status.hpp"

#include "Exception.hpp"
#include "TLS.hpp"

#include <nvcv/Exception.hpp>
#include <util/Assert.h>
#include <util/Status.hpp>

#include <cstring>

namespace nv::cv::priv {

void SetThreadError(std::exception_ptr e)
{
    TLS &tls = GetTLS();

    const int errorMessageLen = sizeof(tls.lastErrorMessage) - 1;

    try
    {
        if (e)
        {
            rethrow_exception(e);
        }
        else
        {
            tls.lastErrorStatus = NVCV_SUCCESS;
            strncpy(tls.lastErrorMessage, "success", errorMessageLen);
        }
    }
    catch (const ::nv::cv::Exception &e)
    {
        tls.lastErrorStatus = NVCV_ERROR_INTERNAL;
        NVCV_ASSERT(!"Exception from public API cannot be originated from internal library implementation");
    }
    catch (const Exception &e)
    {
        tls.lastErrorStatus = e.code();
        strncpy(tls.lastErrorMessage, e.msg(), errorMessageLen);
    }
    catch (const std::invalid_argument &e)
    {
        tls.lastErrorStatus = NVCV_ERROR_INVALID_ARGUMENT;
        strncpy(tls.lastErrorMessage, e.what(), errorMessageLen);
    }
    catch (const std::bad_alloc &)
    {
        tls.lastErrorStatus = NVCV_ERROR_OUT_OF_MEMORY;
        strncpy(tls.lastErrorMessage, "Not enough space for resource allocation", errorMessageLen);
    }
    catch (const std::exception &e)
    {
        tls.lastErrorStatus = NVCV_ERROR_INTERNAL;
        strncpy(tls.lastErrorMessage, e.what(), errorMessageLen);
    }
    catch (...)
    {
        tls.lastErrorStatus = NVCV_ERROR_INTERNAL;
        strncpy(tls.lastErrorMessage, "Unexpected error", errorMessageLen);
    }

    tls.lastErrorMessage[errorMessageLen] = '\0'; // Make sure it's null-terminated
}

NVCVStatus GetLastThreadError() noexcept
{
    return GetLastThreadError(nullptr, 0);
}

NVCVStatus PeekAtLastThreadError() noexcept
{
    return PeekAtLastThreadError(nullptr, 0);
}

NVCVStatus GetLastThreadError(char *outMessage, int outMessageLen) noexcept
{
    NVCVStatus status = PeekAtLastThreadError(outMessage, outMessageLen);

    SetThreadError(std::exception_ptr{});

    return status;
}

NVCVStatus PeekAtLastThreadError(char *outMessage, int outMessageLen) noexcept
{
    TLS &tls = GetTLS();

    if (outMessage != nullptr && outMessageLen > 0)
    {
        strncpy(outMessage, tls.lastErrorMessage, outMessageLen);
        outMessage[outMessageLen - 1] = '\0'; // Make sure it's null-terminated
    }

    return tls.lastErrorStatus;
}

const char *GetName(NVCVStatus status) noexcept
{
    return util::ToString(status);
}

} // namespace nv::cv::priv
