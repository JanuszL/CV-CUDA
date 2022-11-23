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

#include "Export.h"
#include "ImageData.h"
#include "ImageFormat.h"
#include "Status.h"
#include "alloc/Allocator.h"
#include "alloc/Requirements.h"
#include "detail/CudaFwd.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** Underlying image type.
 *
 * Images can have different underlying types depending on the function used to
 * create them.
 * */
typedef enum
{
    /** 2D image. */
    NVCV_TYPE_IMAGE,
    /** Image that wraps an user-allocated image buffer. */
    NVCV_TYPE_IMAGE_WRAPDATA
} NVCVTypeImage;

typedef struct NVCVImage *NVCVImageHandle;

/** Image data cleanup function type */
typedef void (*NVCVImageDataCleanupFunc)(void *ctx, const NVCVImageData *data);

/** Stores the requirements of an image. */
typedef struct NVCVImageRequirementsRec
{
    int32_t         width, height; /*< Image dimensions. */
    NVCVImageFormat format;        /*< Image format. */

    /** Row pitch of each plane, in bytes */
    int32_t planeRowPitchBytes[NVCV_MAX_PLANE_COUNT];

    int32_t          alignBytes; /*< Alignment/block size in bytes */
    NVCVRequirements mem;        /*< Image resource requirements. */
} NVCVImageRequirements;

/** Calculates the resource requirements needed to create an image.
 *
 * @param [in] width,height Image dimensions.
 *                          + Width and height must be > 0.
 *
 * @param [in] format       Image format.
 *                          + Must not be \ref NVCV_IMAGE_FORMAT_NONE.
 *
 * @param [out] reqs        Where the image requirements will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageCalcRequirements(int32_t width, int32_t height, NVCVImageFormat format,
                                                 NVCVImageRequirements *reqs);

/** Constructs and an image instance with given requirements in the given storage.
 *
 * @param [in] reqs Image requirements. Must have been filled in by @ref nvcvImageCalcRequirements.
 *                  + Must not be NULL
 *
 * @param [in] alloc Allocator to be used to allocate needed memory buffers.
 *                   - The following resources are used:
 *                     - host memory: for internal structures.
 *                     - device memory: for image contents buffer.
 *                       If NULL, it'll use the internal default allocator.
 *                   + Allocator must not be destroyed while an image still refers to it.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageConstruct(const NVCVImageRequirements *reqs, NVCVAllocatorHandle alloc,
                                          NVCVImageHandle *handle);

/** Wraps an existing image buffer into an NVCV image instance constructed in given storage
 *
 * It allows for interoperation of external image representations with NVCV.
 * The created image type is \ref NVCV_TYPE_IMAGE_WRAPDATA .
 *
 * @param [in] data Image contents.
 *                  + Must not be NULL
 *                  + Buffer type must not be \ref NVCV_IMAGE_BUFFER_NONE.
 *                  + Image dimensions must be >= 1x1
 *
 * @param [in] cleanup Cleanup function to be called when the image is destroyed
 *                     via @ref nvcvImageDestroy
 *                     If NULL, no cleanup function is defined.
 *
 * @param [in] ctxCleanup Pointer to be passed unchanged to the cleanup function, if defined.
 *
 * @param [out] handle      Where the image instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageWrapDataConstruct(const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup,
                                                  void *ctxCleanup, NVCVImageHandle *handle);

/** Destroys an existing image instance.
 *
 * If the image has type @ref NVCV_TYPE_IMAGE_WRAPDATA and has a cleanup function defined,
 * cleanup will be called.
 *
 * @note The image must not be in use in current and future operations.
 *
 * @param [in] handle Image to be destroyed.
 *                    If NULL, no operation is performed, successfully.
 *                    + The handle must have been created with any of the nvcvImageConstruct functions.
 */
NVCV_PUBLIC void nvcvImageDestroy(NVCVImageHandle handle);

/** Returns the underlying image type.
 *
 * @param [in] handle Image to be queried.
 *                    + Must not be NULL.
 * @param [out] type  The image type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetType(NVCVImageHandle handle, NVCVTypeImage *type);

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
NVCV_PUBLIC NVCVStatus nvcvImageGetSize(NVCVImageHandle handle, int32_t *width, int32_t *height);

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
NVCV_PUBLIC NVCVStatus nvcvImageGetFormat(NVCVImageHandle handle, NVCVImageFormat *fmt);

/**
 * Get the allocator associated with an image.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageGetAllocator(NVCVImageHandle handle, NVCVAllocatorHandle *alloc);

/**
 * Retrieve the image contents.
 *
 * @param[in] handle Image to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] data Where the image buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #VPI_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #VPI_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageExportData(NVCVImageHandle handle, NVCVImageData *data);

#ifdef __cplusplus
}
#endif

#endif // NVCV_IMAGE_H
