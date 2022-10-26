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
 * @file Tensor.h
 *
 * @brief Public C interface to NVCV tensor representation.
 */

#ifndef NVCV_TENSOR_H
#define NVCV_TENSOR_H

#include "Image.h"
#include "ImageFormat.h"
#include "Status.h"
#include "TensorData.h"
#include "alloc/Allocator.h"
#include "alloc/Requirements.h"
#include "detail/CudaFwd.h"
#include "detail/Export.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** Storage for tensor instance. */
typedef struct NVCVTensorStorageRec
{
    /** Instance storage */
    alignas(8) uint8_t storage[1024 + 64];
} NVCVTensorStorage;

typedef struct NVCVTensor *NVCVTensorHandle;

/** Tensor data cleanup function type */
typedef void (*NVCVTensorDataCleanupFunc)(void *ctx, const NVCVTensorData *data);

/** Stores the requirements of an varshape tensor. */
typedef struct NVCVTensorRequirementsRec
{
    /*< Type of each element */
    NVCVPixelType dtype;

    /*< Tensor dimension layout.
     * It's optional. If layout not available, set it to NVCV_TENSOR_NONE. */
    NVCVTensorLayout layout;

    /*< Number of dimensions */
    int32_t ndim;

    /*< Shape of the tensor */
    int64_t shape[NVCV_TENSOR_MAX_NDIM];

    /*< Distance in bytes between each element of a given dimension. */
    int64_t pitchBytes[NVCV_TENSOR_MAX_NDIM];

    /*< Alignment/block size in bytes */
    int32_t alignBytes;

    /*< Tensor resource requirements. */
    NVCVRequirements mem;
} NVCVTensorRequirements;

/** Calculates the resource requirements needed to create a tensor with given shape.
 *
 * @param [in] ndim Number of tensor dimensions.
 *
 * @param [in] shape Pointer to array with tensor shape.
 *                   It must contain at least 'ndim' elements.
 *
 * @param [in] dtype Type of tensor's elements.
 *
 * @param [in] layout Tensor layout.
 *                    Pass NVCV_TENSOR_NONE is layout is not available.
 *                    + Number of dimensions in layout must be @p ndim.
 *
 * @param [out] reqs  Where the tensor requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorCalcRequirements(int32_t ndim, const int64_t *shape, NVCVPixelType dtype,
                                                  NVCVTensorLayout layout, NVCVTensorRequirements *reqs);

/** Calculates the resource requirements needed to create a tensor that holds N images.
 *
 * @param [in] numImages Number of images in the tensor.
 *                       + Must be >= 1.
 *
 * @param [in] width,height Dimensions of each image in the tensor.
 *                          + Must be >= 1x1
 *
 * @param [in] format Format of the images in the tensor.
 *                    + Must not be \ref NVCV_IMAGE_FORMAT_NONE.
 *                    + All planes in must have the same number of channels.
 *                    + No subsampled planes are allowed.
 *
 * @param [out] reqs  Where the tensor requirements will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorCalcRequirementsForImages(int32_t numImages, int32_t width, int32_t height,
                                                           NVCVImageFormat format, NVCVTensorRequirements *reqs);

/** Constructs a tensor instance with given requirements in the given storage.
 *
 * @param [in] reqs Tensor requirements. Must have been filled by one of the nvcvTensorCalcRequirements functions.
 *                  + Must not be NULL
 *
 * @param [in] alloc Allocator to be used to allocate needed memory buffers.
 *                   The following resources are used:
 *                   - device memory
 *                   If NULL, it'll use the internal default allocator.
 *                   + Allocator must not be destroyed while an tensor still refers to it.
 *
 * @param [in,out] storage Memory storage where the tensor instance will be constructed in.
 *
 * @param [out] handle Where the tensor instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the tensor instance.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorConstruct(const NVCVTensorRequirements *reqs, NVCVAllocatorHandle alloc,
                                           NVCVTensorStorage *storage, NVCVTensorHandle *handle);

/** Wraps an existing tensor buffer into an NVCV tensor instance constructed in given storage
 *
 * It allows for interoperation of external tensor representations with NVCV.
 *
 * @param [in] data Tensor contents.
 *                  + Must not be NULL.
 *                  + Allowed buffer types:
 *                    - \ref NVCV_TENSOR_BUFFER_PITCH_DEVICE
 *
 * @param [in] cleanup Cleanup function to be called when the tensor is destroyed
 *                     via @ref nvcvTensorDestroy.
 *                     If NULL, no cleanup function is defined.
 *
 * @param [in] ctxCleanup Pointer to be passed unchanged to the cleanup function, if defined.
 *
 * @param [in,out] storage Memory storage where the tensor instance will be created in.
 *
 * @param [out] handle Where the tensor instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the tensor.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorWrapDataConstruct(const NVCVTensorData *data, NVCVTensorDataCleanupFunc cleanup,
                                                   void *ctxCleanup, NVCVTensorStorage *storage,
                                                   NVCVTensorHandle *handle);

/** Destroys an existing tensor instance.
 *
 * If the tensor is wrapping external data and a cleanup function has been defined, defined,
 * it will be called.
 *
 * @note The tensor object must not be in use in current and future operations.
 *
 * @param [in] handle Tensor to be destroyed.
 *                    If NULL, no operation is performed, successfully.
 *                    + The handle must have been created with any of the nvcvTensorConstruct functions.
 */
NVCV_PUBLIC void nvcvTensorDestroy(NVCVTensorHandle handle);

/**
 * Get the type of the tensor elements (its data type).
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] type Where the type will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetDataType(NVCVTensorHandle handle, NVCVPixelType *type);

/**
 * Get the tensor layout
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] layout Where the tensor layout will be written to.
 *                    + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetLayout(NVCVTensorHandle handle, NVCVTensorLayout *layout);

/**
 * Get the allocator associated with the tensor.
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] alloc Where the allocator handle will be written to.
 *                   + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetAllocator(NVCVTensorHandle handle, NVCVAllocatorHandle *alloc);

/**
 * Retrieve the tensor contents.
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *
 * @param[out] data Where the tensor buffer information will be written to.
 *                  + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorExportData(NVCVTensorHandle handle, NVCVTensorData *data);

/**
 * Retrieve the tensor shape.
 *
 * @param[in] handle Tensor to be queried.
 *                   + Must not be NULL.
 *                   + Must have been created by @ref nvcvTensorConstruct.
 *
 * @param[in,out] ndim Number of elements in output shape buffer.
 *                     When function returns, it stores the actual number of dimensions in the tensor.
 *                     Set it to NVCV_TENSOR_MAX_NDIM to return the full shape in @shape.
 *                     Set it to 0 if only tensor's ndim must be returned.
 *
 * @param[out] shape Where the tensor shape will be written to.
 *                   Must point to a buffer with @p ndim elements.
 *                   Elements above actual number of dimensions will be set to 1.
 *                   + If NULL, @p ndim must be 0.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorGetShape(NVCVTensorHandle handle, int32_t *ndim, int64_t *shape);

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSOR_H
