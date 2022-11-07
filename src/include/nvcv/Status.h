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
 * @file Status.h
 *
 * @brief Declaration of NVCV C status codes handling functions.
 */

#ifndef NVCV_STATUS_H
#define NVCV_STATUS_H

#include "detail/Export.h"

#include <stdint.h>

/**
 * @brief Declares entities to handle return status codes used in NVCV.
 *
 * NVCV functions uses status codes to return if they succeeded or not.
 * @defgroup NVCV_C_UTIL_STATUS Status Codes
 * @{
 */

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Status codes.
 */
typedef enum
{
    NVCV_SUCCESS = 0,                /**< Operation completed successfully. */
    NVCV_ERROR_NOT_IMPLEMENTED,      /**< Operation isn't implemented. */
    NVCV_ERROR_INVALID_ARGUMENT,     /**< Invalid argument, either wrong range or value not accepted. */
    NVCV_ERROR_INVALID_IMAGE_FORMAT, /**< Image type not accepted. */
    NVCV_ERROR_INVALID_OPERATION,    /**< Operation isn't valid in this context. */
    NVCV_ERROR_DEVICE,               /**< Device backend error. */
    NVCV_ERROR_NOT_READY,            /**< Operation not completed yet, try again later. */
    NVCV_ERROR_OUT_OF_MEMORY,        /**< Not enough free memory to allocate object. */
    NVCV_ERROR_INTERNAL,             /**< Internal, non specific error. */
    NVCV_ERROR_NOT_COMPATIBLE,       /**< Implementation version incompatibility. */
    NVCV_ERROR_OVERFLOW,             /**< Result is larger than maximum accepted value. */
    NVCV_ERROR_UNDERFLOW,            /**< Result is smaller than minimum accepted value. */
} NVCVStatus;

/**
 * @brief Maximum status message length in bytes.
 *
 * This is the maximum number of bytes that will be written by \ref
 * nvcvGetLastStatusMessage and \ref nvcvPeekAtLastStatusMessage to the status
 * message output buffer. It includes the final '\0'.
 */
#define NVCV_MAX_STATUS_MESSAGE_LENGTH 256

/**
 * @brief Returns a string representation of the status code.
 *
 * @param [in] code Status code whose string representation is to be returned.
 *
 * @returns The string representation of the status code.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvStatusGetName(NVCVStatus code);

/**
 * @brief Returns and resets the error status of the last CV-CUDA function call that failed in current thread.
 *
 * A new call to this function will return \ref NVCV_SUCCESS, as the thread-specific
 * status was reset. This operation doesn't affect the statuses in other threads.
 *
 * @returns The status of the last CV-CUDA function call that failed in current thread.
 */
NVCV_PUBLIC NVCVStatus nvcvGetLastError();

/**
 * @brief Returns and resets the error status code and message of the last CV-CUDA function call that failed in current thread.
 *
 * A new call to this function will return \ref NVCV_SUCCESS, as the thread-specific
 * status was reset. This operation doesn't affect the status in other threads.
 *
 * It's guaranteed that the message is never larger than
 * \ref NVCV_MAX_STATUS_MESSAGE_LENGTH bytes, including the '\0' string terminator.
 *
 * @param[out] msgBuffer Pointer to memory where the status message will be written to.
 *                       If NULL, no message is returned.
 *
 * @param[in] lenBuffer Size in bytes of msgBuffer.
 *                      + If less than zero, \p lenBuffer is assumed to be 0.
 *
 * @returns The status of the last CV-CUDA function call that failed in current thread.
 */
NVCV_PUBLIC NVCVStatus nvcvGetLastErrorMessage(char *msgBuffer, int32_t lenBuffer);

/**
 * @brief Returns the error status of the last CV-CUDA function call that failed in current thread.
 *
 * The internal status code and message of current thread won't be reset.
 *
 * @returns The status of the last CV-CUDA function call that failed in current thread.
 */
NVCV_PUBLIC NVCVStatus nvcvPeekAtLastError();

/**
 * @brief Returns the status code and message of the last CV-CUDA function call that failed in current thread.
 *
 * The internal status code and message of current thread won't be reset.
 *
 * It's guaranteed that the message is never larger than
 * \ref NVCV_MAX_STATUS_MESSAGE_LENGTH bytes, including the '\0' string terminator.
 *
 * @param[out] msgBuffer Pointer to memory where the status message will be written to.
 *                       If NULL, no message is returned.
 *
 * @param[in] lenBuffer Size in bytes of msgBuffer.
 *                      + If less than zero, lenBuffer is assumed to be 0.
 *
 * @returns The status of the last CV-CUDA function call that failed in current thread.
 */
NVCV_PUBLIC NVCVStatus nvcvPeekAtLastErrorMessage(char *msgBuffer, int32_t lenBuffer);

/**
 * @brief Sets the internal status in current thread.
 *
 * This is used by nvcv extensions and/or language bindings to seamlessly
 * integrate their status handling with the C API.
 *
 * @param[in] status The status code to be set
 * @param[in] msg The status message associated with the status code.
 *                Pass NULL if no customized error message is needed
 */
NVCV_PUBLIC void nvcvSetThreadStatus(NVCVStatus status, const char *msg);

/**@}*/

#ifdef __cplusplus
}
#endif

#endif // NVCV_STATUS_H
