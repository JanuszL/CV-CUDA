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
 * @file Config.h
 *
 * @brief Public C interface to NVCV configuration.
 */

#ifndef NVCV_CONFIG_H
#define NVCV_CONFIG_H

#include "Export.h"
#include "Status.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Set a hard limit on the number of image handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of image handles in the future.
 *
 * @param[in] maxCount Maximum number of image handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no image handles created and not destroyed.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxImageCount(int32_t maxCount);

/**
 * Set a hard limit on the number of image batch handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of image batch handles in the future.
 *
 * @param[in] maxCount Maximum number of image batch handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no image batch handles created and not destroyed.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxImageBatchCount(int32_t maxCount);

/**
 * Set a hard limit on the number of tensor handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of tensor handles in the future.
 *
 * @param[in] maxCount Maximum number of tensor handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no tensor handles created and not destroyed.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxTensorCount(int32_t maxCount);

/**
 * Set a hard limit on the number of allocator handles that can be created.
 *
 * The function will preallocate all resources necessary to satisfy creation
 * of a limited number of allocator handles in the future.
 *
 * @param[in] maxCount Maximum number of allocator handles that can be created.
 *                     If negative, switches to dynamic allocation, no hard limit is defined.
 *                     + There must be no allocator handles created and not destroyed.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvConfigSetMaxAllocatorCount(int32_t maxCount);

#ifdef __cplusplus
}
#endif

#endif // NVCV_CONFIG_H
