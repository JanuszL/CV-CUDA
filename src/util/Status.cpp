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

namespace nv::cv::util {

/**
 * Convert an error value to a string
 */
const char *ToString(NVCVStatus err) noexcept
{
#define CASE(ERR) \
    case ERR:     \
        return #ERR

    // written this way, without a default case,
    // the compiler can warn us if we forgot to add a new error here.
    switch (err)
    {
        CASE(NVCV_SUCCESS);
        CASE(NVCV_ERROR_NOT_IMPLEMENTED);
        CASE(NVCV_ERROR_INVALID_ARGUMENT);
        CASE(NVCV_ERROR_INVALID_IMAGE_FORMAT);
        CASE(NVCV_ERROR_INVALID_OPERATION);
        CASE(NVCV_ERROR_DEVICE);
        CASE(NVCV_ERROR_NOT_READY);
        CASE(NVCV_ERROR_OUT_OF_MEMORY);
        CASE(NVCV_ERROR_INTERNAL);
    }

    // Status not found?
    return "Unknown error";
#undef CASE
}

} // namespace nv::cv::util
