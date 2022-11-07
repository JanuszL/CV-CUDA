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

#include "TLS.hpp"

#include <nvcv/Exception.hpp>
#include <util/Assert.h>
#include <util/Exception.hpp>
#include <util/Status.hpp>

#include <cstring>

namespace nv::cv::priv {

void SetThreadError(std::exception_ptr e)
{
    GetTLS().lastError = e;
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

    // Clear the thread error state
    GetTLS().lastError = {};

    return status;
}

NVCVStatus PeekAtLastThreadError(char *outMessage, int outMessageLen) noexcept
{
    NVCVStatus status;
    try
    {
        TLS &tls = GetTLS();

        if (tls.lastError)
        {
            rethrow_exception(tls.lastError);
        }
        else
        {
            status = NVCV_SUCCESS;
            if (outMessage != nullptr)
            {
                strncpy(outMessage, "success", outMessageLen);
            }
        }
    }
    catch (const ::nv::cv::Exception &e)
    {
        status = NVCV_ERROR_INTERNAL;
        NVCV_ASSERT(!"Eception from public API cannot be originated from internal library implementation");
    }
    catch (const util::Exception &e)
    {
        status = e.code();
        if (outMessage != nullptr)
        {
            strncpy(outMessage, e.msg(), outMessageLen);
        }
    }
    catch (const std::invalid_argument &e)
    {
        status = NVCV_ERROR_INVALID_ARGUMENT;
        if (outMessage != nullptr)
        {
            strncpy(outMessage, e.what(), outMessageLen);
        }
    }
    catch (const std::bad_alloc &)
    {
        status = NVCV_ERROR_OUT_OF_MEMORY;
        if (outMessage != nullptr)
        {
            strncpy(outMessage, "Not enough space for resource allocation", outMessageLen);
        }
    }
    catch (const std::exception &e)
    {
        status = NVCV_ERROR_INTERNAL;
        if (outMessage != nullptr)
        {
            strncpy(outMessage, e.what(), outMessageLen);
        }
    }
    catch (...)
    {
        status = NVCV_ERROR_INTERNAL;
        if (outMessage != nullptr)
        {
            strncpy(outMessage, "Unexpected error", outMessageLen);
        }
    }

    if (outMessage && outMessageLen > 0)
    {
        outMessage[outMessageLen - 1] = '\0'; // Make sure it's null-terminated
    }

    return status;
}

const char *GetName(NVCVStatus status) noexcept
{
    return util::ToString(status);
}

} // namespace nv::cv::priv
