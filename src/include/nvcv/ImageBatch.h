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
 * @file ImageBatch.h
 *
 * @brief Public C interface to NVCV image batch representation.
 */

#ifndef NVCV_IMAGEBATCH_H
#define NVCV_IMAGEBATCH_H

#include "Image.h"
#include "ImageBatchData.h"
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

/** Underlying image batch type. */
typedef enum
{
    /** Batch of 2D images of different dimensions. */
    NVCV_TYPE_IMAGEBATCH_VARSHAPE,
    /** Batch of 2D images that have the same dimensions. */
    NVCV_TYPE_IMAGEBATCH_TENSOR,
    /** Image batch that wraps an user-allocated tensor buffer. */
    NVCV_TYPE_IMAGEBATCH_TENSOR_WRAPDATA,
} NVCVTypeImageBatch;

/** Storage for image batch instance. */
typedef struct NVCVImageBatchStorageRec
{
    /** Instance storage */
    alignas(8) uint8_t storage[1024];
} NVCVImageBatchStorage;

typedef struct NVCVImageBatch *NVCVImageBatchHandle;

/** Image batch data cleanup function type */
typedef void (*NVCVImageBatchDataCleanupFunc)(void *ctx, const NVCVImageBatchData *data);

/** Stores the requirements of an varshape image batch. */
typedef struct NVCVImageBatchVarShapeRequirementsRec
{
    int32_t         capacity; /*< Maximum number of images stored. */
    NVCVImageFormat format;   /*< Format of batched images. */

    int32_t          alignBytes; /*< Alignment/block size in bytes */
    NVCVRequirements mem;        /*< Image batch resource requirements. */
} NVCVImageBatchVarShapeRequirements;

/** Calculates the resource requirements needed to create a varshape image batch.
 *
 * @param [in] capacity Maximum number of images that fits in the image batch.
 *                      + Must be >= 1.
 *
 * @param [in] format Format of the images in the image batch.
 *                    + Must not be \ref NVCV_IMAGE_FORMAT_NONE.
 *
 * @param [out] reqs  Where the image batch requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeCalcRequirements(int32_t capacity, NVCVImageFormat format,
                                                              NVCVImageBatchVarShapeRequirements *reqs);

/** Constructs a varshape image batch instance with given requirements in the given storage.
 *
 * @param [in] reqs Image batch requirements. Must have been filled in by @ref nvcvImageBatchVarShapeCalcRequirements.
 *                  + Must not be NULL
 *
 * @param [in] alloc        Allocator to be used to allocate needed memory buffers.
 *                          The following resources are used:
 *                          - host memory
 *                          - device memory
 *                          If NULL, it'll use the internal default allocator.
 *                          + Allocator must not be destroyed while an image batch still refers to it.
 *
 * @param [in,out] storage Memory storage where the image batch instance will be constructed in.
 *
 * @param [out] handle      Where the image batch instance handle will be written to.
 *                          + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the image batch instance.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeConstruct(const NVCVImageBatchVarShapeRequirements *reqs,
                                                       NVCVAllocatorHandle alloc, NVCVImageBatchStorage *storage,
                                                       NVCVImageBatchHandle *handle);

/** Destroys an existing image batch instance.
 *
 * If the image has type @ref NVCV_TYPE_IMAGEBATCH_TENSOR_WRAPDATA and has a cleanup function defined,
 * cleanup will be called.
 *
 * @note The image batch object must not be in use in current and future operations.
 *
 * @param [in] handle Image batch to be destroyed.
 *                    If NULL, no operation is performed, successfully.
 *                    + The handle must have been created with any of the nvcvImageBatchXXXConstruct functions.
 */
NVCV_PUBLIC void nvcvImageBatchDestroy(NVCVImageBatchHandle handle);

/** Returns the underlying type of the image batch.
 *
 * @param [in] handle Image batch to be queried.
 *                    + Must not be NULL.
 * @param [out] type  The image batch type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetType(NVCVImageBatchHandle handle, NVCVTypeImageBatch *type);

/** Returns the capacity of the image batch.
 *
 * @param [in] handle Image batch to be queried.
 *                    + Must not be NULL.
 * @param [out] capacity  The capacity of the given image batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetCapacity(NVCVImageBatchHandle handle, int32_t *capacity);

/**
 * Get the format of the images the image batch can store.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] format Where the image format will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetFormat(NVCVImageBatchHandle handle, NVCVImageFormat *fmt);

/**
 * Get the allocator associated with an image batch.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetAllocator(NVCVImageBatchHandle handle, NVCVAllocatorHandle *alloc);

/**
 * Retrieve the number of images in the batch.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] numImages Where the number of images will be written to.
 *                       + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchGetSize(NVCVImageBatchHandle handle, int32_t *numImages);

/**
 * Retrieve the image batch contents.
 *
 * @param[in] handle Image batch to be queried.
 *                   + Must not be NULL.
 *
 * @param[in] stream CUDA stream where the export operation will execute.
 *
 * @param[out] data Where the image batch buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchExportData(NVCVImageBatchHandle handle, CUstream stream, NVCVImageBatchData *data);

/**
 * Push images to the end of the image batch.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] images Pointer to a buffer with the image handles to be added.
 *                   + Must not be NULL.
 *                   + Must point to an array of at least @p numImages image handles.
 *                   + The images must not be destroyed while they're being referenced by the image batch.
 *
 * @param[in] numImages Number of images in the @p images array.
 *                      + Must be >= 1.
 *                      + Final number of images must not exceed the image batch capacity.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OVERFLOW         Image batch capacity exceeded.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapePushImages(NVCVImageBatchHandle handle, const NVCVImageHandle *images,
                                                        int32_t numImages);

/**
 * Callback function used to push images.
 *
 * Every time it is called, it'll return the next image in a sequence.
 * It'll return NULL after the last image is returned.
 *
 * @param [in] ctx User context passed by the user.
 * @returns The next NVCVImage handle in the sequence, or NULL if there's no more
 *          images to return.
 */
typedef NVCVImageHandle (*NVCVPushImageFunc)(void *ctx);

/**
 * Push to the end of the batch the images returned from a callback function.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] cbGetImage Function that returns each image that is pushed to the batch.
 *                       It'll keep being called until it return NULL, meaning that there are no more
 *                       images to be returned.
 *                       + Must not be NULL.
 *                       + It must return NULL before the capacity of the batch is exceeded.
 *                       + The images returned must not be destroyed while they're being referenced by the image batch.
 *
 * @param[in] ctxCallback Pointer passed to the callback function unchanged.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OVERFLOW         Image batch capacity exceeded.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapePushImagesCallback(NVCVImageBatchHandle handle,
                                                                NVCVPushImageFunc cbPushImage, void *ctxCallback);

/**
 * Pop images from the end of the image batch.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] numImages Number of images in the @p images array.
 *                      + Must be >= 1.
 *                      + Must be <= number of images in the batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_UNDERFLOW        Tried to remove more images that there are in the batch.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapePopImages(NVCVImageBatchHandle handle, int32_t numImages);

/**
 * Clear the contents of the varshape image batch.
 *
 * It sets its size to 0.
 *
 * @param[in] handle Image batch to be manipulated
 *                   + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeClear(NVCVImageBatchHandle handle);

/**
 * Retrieve the image handles from the varshape image batch.
 *
 * @param[in] handle Varshape image batch to be queried
 *                   + Must not be NULL.
 *                   + The handle must have been created with @ref nvcvImageBatchVarShapeConstruct.
 *
 * @param[in] begOffset Index offset of the first image to be retrieved.
 *                      To retrieve starting from the first image, pass 0.
 *                      + Must be < number of images in the batch.
 *
 * @param[out] outBuffer Where the image handles will be written to.
 *
 * @param[in] numImages Number of images to be retrieved.
 *                      + Must be >= 0.
 *                      + Must be begOffset+numImages <= number of images in the batch.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_ERROR_OVERFLOW         Tried to retrieve more images that there are in the batch.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvImageBatchVarShapeGetImages(NVCVImageBatchHandle handle, int32_t begIndex,
                                                       NVCVImageHandle *outImages, int32_t numImages);

#ifdef __cplusplus
}
#endif

#endif // NVCV_IMAGEBATCH_H