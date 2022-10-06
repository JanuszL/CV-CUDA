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
#include "alloc/Requirements.h"
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
    /** Image that wraps an user-allocated image buffer. */
    NVCV_TYPE_IMAGE_WRAP_DATA
} NVCVTypeImage;

/** Handle to an image instance. */
typedef struct NVCVImageImpl *NVCVImage;

/** Image data cleanup function type */
typedef void (*NVCVImageDataCleanupFunc)(void *ctx, const NVCVImageData *data);

/** Stores the requirements of an image. */
typedef struct NVCVImageRequirementsRec
{
    int32_t         width, height; /*< Image dimensions. */
    NVCVImageFormat format;        /*< Image format. */

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
 * @param [out] handle      Where the image instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageCalcRequirements(int32_t width, int32_t height, NVCVImageFormat format,
                                                 NVCVImageRequirements *reqs);

/** Creates and allocates an image instance given its requirements.
 *
 * @param [in] reqs Image requirements. Must have been filled in by @ref nvcvImageGatherRequirements.
 *                  + Must not be NULL
 *
 * @param [in] alloc        Allocator to be used to allocate needed memory buffers.
 *                          The following resources are used:
 *                          - host memory: for internal structures.
 *                          - device memory: for image contents buffer.
 *                          If NULL, it'll use the internal default allocator.
 *                          + Allocator must not be destroyed while an image still refers to it.
 *
 * @param [out] handle      Where the image instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageCreate(const NVCVImageRequirements *reqs, NVCVAllocator *alloc, NVCVImage *handle);

/** Wraps an existing image buffer into an NVCV image instance.
 *
 * It allows for interoperation of external image representations with NVCV.
 * The created image type is \ref NVCV_TYPE_IMAGE_WRAP_DATA .
 *
 * If the image doesn't refer to a buffer, it's considered empty. In this case, it's dimensions is 0x0,
 * and image foramt is \ref NVCV_IMAGE_FORMAT_NONE .
 *
 * The image buffer can be redefined by \ref nvcvImageWrapResetData, or reset by
 * \ref nvcvImageWrapResetData .
 *
 * @param [in] data Image contents.
 *                  If NULL, the created image is empty.
 *                  When not NULL, the buffer ownership isn't transferred. It
 *                  must not be destroyed while the NVCV image still refers to
 *                  it.
 *                  + When not NULL, buffer type must not be \ref NVCV_IMAGE_BUFFER_NONE.
 *
 * @param [in] cleanup Cleanup function to be called when an not empty image is destroyed
 *                     via @ref nvcvImageDestroy, or the image data is reset via
 *                     @ref nvcvImageResetData.
 *                     If NULL, no cleanup function is defined.
 *
 * @param [in] ctxCleanup Pointer to be passed unchanged to the cleanup function, if defined.
 *
 * @param [in] alloc        Allocator to be used to allocate needed memory buffers.
 *                          The following resources are used:
 *                          - host memory: for internal structures.
 *                          If NULL, it'll use the internal default allocator.
 *                          + Allocator must not be destroyed while an image still refers to it.
 *
 * @param [out] handle      Where the image instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageCreateWrapData(const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup,
                                               void *ctxCleanup, NVCVAllocator *alloc, NVCVImage *handle);

/** Destroys an existing image instance.
 *
 * If the image has type @ref NVCV_TYPE_IMAGE_WRAP_DATA, is not empty and has a cleanup function defined,
 * cleanup will be called.
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
NVCV_PUBLIC NVCVStatus nvcvImageGetAllocator(NVCVImage handle, NVCVAllocator **alloc);

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

/** Redefines the buffer pointed by the image wrap object, leaving cleanup function intact.
 *
 * If the image currently has a cleanup function defined and the image is not empty,
 * this cleanup will be called on the old image data. The cleanup function won't be changed.
 *
 * @param [in] handle Handle to the image to have its buffer replaced.
 *                    + The image type must be @ref NVCV_TYPE_IMAGE_WRAP_DATA
 *
 * @param [in] data New image contents.
 *                  If NULL, the image will be set to empty.
 *                  When not NULL, the buffer ownership isn't transferred automatically.
 *                  If needed, ownership transfer can be implemted via cleanup function,
 *                  or else ot must not be destroyed while the NVCV image still refers to it.
 *                  must not be destroyed while the NVCV image still refers to
 *                  it.
 *                  + When not NULL, buffer type must not be \ref NVCV_IMAGE_BUFFER_NONE.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageWrapResetData(NVCVImage handle, const NVCVImageData *data);

/** Redefines the buffer pointed by the image wrap object and its cleanup function.
 *
 * If the image currently has a cleanup function defined and the image is not empty,
 * this old cleanup will be called on the old image data prior to setting them to the
 * new values.
 *
 * @param [in] handle Handle to the image to have its buffer replaced.
 *                    + The image type must be @ref NVCV_TYPE_IMAGE_WRAP_DATA
 *
 * @param [in] data New image contents.
 *                  If NULL, the image will be set to empty.
 *                  When not NULL, the buffer ownership isn't automatically transferred.
 *                  If needed, ownership transfer can be implemted via cleanup function,
 *                  or else ot must not be destroyed while the NVCV image still refers to it.
 *                  + When not NULL, buffer type must not be \ref NVCV_IMAGE_BUFFER_NONE.
 *
 * @param [in] cleanup Replaces the existing cleanup function.
 *                     Pass NULL to undefine it.
 *
 * @param [in] ctxCleanup Pointer to be passed unchanged to the cleanup function, if defined.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageWrapResetDataAndCleanup(NVCVImage handle, const NVCVImageData *data,
                                                        NVCVImageDataCleanupFunc cleanup, void *ctxCleanup);

#ifdef __cplusplus
}
#endif

#endif // NVCV_IMAGE_H
