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

#ifndef NVCV_TENSORDATA_H
#define NVCV_TENSORDATA_H

#include "ImageFormat.h"
#include "TensorLayout.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef NVCVPixelType NVCVElementType;

/** Stores the tensor plane contents. */
typedef struct NVCVTensorBufferPitchRec
{
    int64_t pitchBytes[NVCV_TENSOR_MAX_NDIM];

    /** Pointer to memory buffer with tensor contents.
     * Pixel with type T is addressed by:
     * pixAttr = (uint8_t *)mem + shape[0]*pitchBytes[0] + ... + shape[ndim-1]*pitchBytes[ndim-1];
     */
    void *data;
} NVCVTensorBufferPitch;

/** Represents how the image buffer data is stored. */
typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    NVCV_TENSOR_BUFFER_NONE = 0,

    /** GPU-accessible with equal-shape planes in pitch-linear layout. */
    NVCV_TENSOR_BUFFER_PITCH_DEVICE,
} NVCVTensorBufferType;

/** Represents the available methods to access image batch contents.
 * The correct method depends on \ref NVCVTensorData::bufferType. */
typedef union NVCVTensorBufferRec
{
    /** Tensor image batch stored in pitch-linear layout.
     * To be used when \ref NVCVTensorData::bufferType is:
     * - \ref NVCV_TENSOR_BUFFER_PITCH_DEVICE
     */
    NVCVTensorBufferPitch pitch;
} NVCVTensorBuffer;

/** Stores information about image batch characteristics and content. */
typedef struct NVCVTensorDataRec
{
    NVCVElementType  dtype;
    NVCVTensorLayout layout;

    int32_t ndim;
    int64_t shape[NVCV_TENSOR_MAX_NDIM];

    /** Type of image batch buffer.
     *  It defines which member of the \ref NVCVTensorBuffer tagged union that
     *  must be used to access the image batch buffer contents. */
    NVCVTensorBufferType bufferType;

    /** Stores the image batch contents. */
    NVCVTensorBuffer buffer;
} NVCVTensorData;

/**
 * Permute the input shape with given layout to a different layout.
 *
 * @param[in] srcLayout The layout of the source shape.
 *
 * @param[in] srcShape The shape to be permuted.
 *                     + Must not be NULL.
 *                     + Number of dimensions must be equal to dimensions in @p srcLayout
 *
 * @param[in] dstLayout The layout of the destination shape.
 *
 * @param[out] dstShape Where the permutation will be written to.
 *                      + Must not be NULL.
 *                      + Number of dimensions must be equal to dimensions in @p dstLayout
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorShapePermute(NVCVTensorLayout srcLayout, const int64_t *srcShape,
                                              NVCVTensorLayout dstLayout, int64_t *dstShape);

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORDATA_H
