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

/**
 * @file Status.hpp
 *
 * @brief Declaration of NVCV C++ status codes handling functions.
 */

#ifndef NVCV_STATUS_HPP
#define NVCV_STATUS_HPP

#include "Status.h"

#include <cstdint>
#include <ostream>

namespace nv { namespace cv {

/**
 * @brief Status codes.
 *
 * This enum is coupled to NVCVStatus, the status codes are the same.
 * For further details, see \ref NVCVStatus.
 * @defgroup NVCV_CPP_UTIL_STATUS Status Codes
 * @{
 */
enum class Status : int8_t
{
    SUCCESS                    = NVCV_SUCCESS,
    ERROR_NOT_IMPLEMENTED      = NVCV_ERROR_NOT_IMPLEMENTED,
    ERROR_INVALID_ARGUMENT     = NVCV_ERROR_INVALID_ARGUMENT,
    ERROR_INVALID_IMAGE_FORMAT = NVCV_ERROR_INVALID_IMAGE_FORMAT,
    ERROR_INVALID_OPERATION    = NVCV_ERROR_INVALID_OPERATION,
    ERROR_DEVICE               = NVCV_ERROR_DEVICE,
    ERROR_NOT_READY            = NVCV_ERROR_NOT_READY,
    ERROR_OUT_OF_MEMORY        = NVCV_ERROR_OUT_OF_MEMORY,
    ERROR_INTERNAL             = NVCV_ERROR_INTERNAL,
    ERROR_NOT_COMPATIBLE       = NVCV_ERROR_NOT_COMPATIBLE,
    ERROR_OVERFLOW             = NVCV_ERROR_OVERFLOW,
    ERROR_UNDERFLOW            = NVCV_ERROR_UNDERFLOW
};

inline const char *GetName(Status status)
{
    return nvcvStatusGetName(static_cast<NVCVStatus>(status));
}

inline std::ostream &operator<<(std::ostream &out, Status status)
{
    return out << static_cast<NVCVStatus>(status);
}

inline std::ostream &operator<<(std::ostream &out, NVCVStatus status)
{
    return out << nvcvStatusGetName(status);
}

/**@}*/

}} // namespace nv::cv

#endif // NVCV_STATUS_HPP
