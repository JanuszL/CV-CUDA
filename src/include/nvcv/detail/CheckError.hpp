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

#ifndef NVCV_DETAIL_CHECKERROR_HPP
#define NVCV_DETAIL_CHECKERROR_HPP

#include "../Exception.hpp"
#include "../Status.h"

#include <cassert>

namespace nv::cv::detail {

inline void CheckThrow(NVCVStatus status)
{
    if (status != NVCV_SUCCESS)
    {
        char msg[NVCV_MAX_STATUS_MESSAGE_LENGTH];

        NVCVStatus tmp = nvcvGetLastStatusMessage(msg, sizeof(msg));
        (void)tmp;
        assert(tmp == status);

        throw Exception(static_cast<Status>(status), msg);
    }
}

} // namespace nv::cv::detail

#endif // NVCV_DETAIL_CHECKERROR_HPP
