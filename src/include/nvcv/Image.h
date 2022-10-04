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
 * @file Image.h
 *
 * @brief Public C interface to NVCV image representation.
 */

#ifndef NVCV_IMAGE_H
#define NVCV_IMAGE_H

#include "ImageData.h"
#include "ImageFormat.h"
#include "Status.h"
#include "alloc/Allocator.h"
#include "detail/CudaFwd.h"
#include "detail/Export.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** Underlying image type.
 *
 * Images can have different underlying types depending on the function used to
 * create them. Some functions expect an image of some type, for instance, images
 * that wrap an user-allocated image allow users to redefine the image buffer after
 * the image is allocated, or reset it altogether. These operations aren't available
 * to other image types, and calling these function on them will result in an error.
 * */
typedef enum
{
    /** 2D image. */
    NVCV_TYPE_IMAGE,
} NVCVTypeImage;

/** Handle to an image instance. */
typedef struct NVCVImageImpl *NVCVImage;

/** Creates and allocates an image instance.
 *
 * @param [in] width,height Image dimensions.
 *                          + Width and height must be > 0.
 * @param [in] fmt          Image format.
 *                          + Must not be \ref NVCV_IMAGE_FORMAT_NONE.
 * @param [in] alloc        Allocator to be used to allocate needed memory buffers.
 *                          The following resources are used:
 *                          - host memory: for internal structures.
 *                          - device memory: for image contents buffer.
 *                          If NULL, it'll use the internal default allocator.
 *                          + Allocator must not be destroyed while an image still refers to it.
 * @param [out] handle      Where the image instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageCreate(int32_t width, int32_t height, NVCVImageFormat fmt, NVCVAllocator alloc,
                                       NVCVImage *handle);

/** Destroys an existing image instance.
 *
 * @note The image must not be in use in current and future operations.
 *
 * @param [in] handle Image to be destroyed.
 *                    If NULL, no operation is performed, successfully.
 *                    + The handle must have been created with any of the nvcvImageCreate functions.
 */
NVCV_PUBLIC void nvcvImageDestroy(NVCVImage handle);

/** Returns the underlying image type.
 *
 * @param [in] handle Image to be queried.
 *                    + Must not be NULL.
 * @param [out] type  The image type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetType(NVCVImage handle, NVCVTypeImage *type);

/**
 * Get the image dimensions in pixels.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] width, height Where dimensions will be written to.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetSize(NVCVImage handle, int32_t *width, int32_t *height);

/**
 * Get the image format.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] format Where the image format will be written to.
 *                    + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetFormat(NVCVImage handle, NVCVImageFormat *fmt);

/**
 * Get the allocator associated with an image.
 *
 * @param[in] handle Image to be queried.
 *                + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetAllocator(NVCVImage handle, NVCVAllocator *alloc);

/**
 * Retrieve the image contents.
 *
 * @param[in] handle Image to be queried.
 *                + Must not be NULL.
 *
 * @param[out] data Where the image buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageExportData(NVCVImage handle, NVCVImageData *data);

#ifdef __cplusplus
}
#endif

#endif // NVCV_IMAGE_H
