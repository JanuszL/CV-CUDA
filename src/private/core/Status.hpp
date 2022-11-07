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

#ifndef NVCV_PRIV_STATUS_HPP
#define NVCV_PRIV_STATUS_HPP

#include <nvcv/Status.h>

#include <exception>
#include <iosfwd>

namespace nv::cv::priv {

NVCVStatus GetLastThreadError(char *outMessage, int outMessageLen) noexcept;
NVCVStatus PeekAtLastThreadError(char *outMessage, int outMessageLen) noexcept;

NVCVStatus GetLastThreadError() noexcept;
NVCVStatus PeekAtLastThreadError() noexcept;

const char *GetName(NVCVStatus status) noexcept;

void SetThreadError(std::exception_ptr e);

template<class F>
NVCVStatus ProtectCall(F &&fn)
{
    try
    {
        fn();
        return NVCV_SUCCESS;
    }
    catch (...)
    {
        SetThreadError(std::current_exception());
        return PeekAtLastThreadError();
    }
}

} // namespace nv::cv::priv

#endif // NVCV_PRIV_STATUS_HPP
