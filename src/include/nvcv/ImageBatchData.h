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

#ifndef NVCV_IMAGEBATCHDATA_H
#define NVCV_IMAGEBATCHDATA_H

#include "ImageData.h"
#include "ImageFormat.h"

/** Stores the image plane in a variable shape image batch. */
typedef struct NVCVImageBatchVarShapeBufferPitchRec
{
    /** Union of all image dimensions.
     * If 0 and number of images is >= 1, this value
     * must not be relied upon. */
    int32_t maxWidth, maxHeight;

    /** Pointer to all image planes in pitch-linear layout in the image batch.
     * It's an array of `numPlanesPerImage*numImages` planes. The number of planes
     * in the image can be fetched from the image batch's format. With that,
     * plane P of image N can be indexed as imgPlanes[N*numPlanesPerImage + P].
     */
    NVCVImagePlanePitch *imgPlanes;
} NVCVImageBatchVarShapeBufferPitch;

/** Stores the tensor plane contents. */
typedef struct NVCVImageBatchTensorBufferPitchRec
{
    /** Number of images in the batch.
     *  + Must be >= 1. */
    int32_t numImages;

    /** Distance in bytes from beginning of first plane of one image to the
     *  first plane of the next image.
     *  + Must be >= 1. */
    int64_t imgPitchBytes;

    /** Distance in bytes from beginning of one row to the next.
     *  + Must be >= 1. */
    int32_t rowPitchBytes;

    /** Dimensions of each image.
     * + Must be >= 1x1 */
    int32_t imgWidth, imgHeight;

    /** Buffer of all image planes in pitch-linear layout.
     *  It assumes all planes have same dimension specified by imgWidth/imgHeight,
     *  and that all planes have the same row pitch.
     *  + Only the first N elements must have valid data, where N is the number of planes
     *    defined by @ref NVCVImageBatchData::format. */
    void *planeBuffer[NVCV_MAX_PLANE_COUNT];
} NVCVImageBatchTensorBufferPitch;

/** Represents how the image buffer data is stored. */
typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    NVCV_IMAGE_BATCH_BUFFER_NONE = 0,

    /** GPU-accessible with variable-shape planes in pitch-linear layout. */
    NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_PITCH_DEVICE,
} NVCVImageBatchBufferType;

/** Represents the available methods to access image batch contents.
 * The correct method depends on \ref NVCVImageBatchData::bufferType. */
typedef union NVCVImageBatchBufferRec
{
    /** Varshape image batch stored in pitch-linear layout.
     * To be used when \ref NVCVImageBatchData::bufferType is:
     * - \ref NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_PITCH_DEVICE
     */
    NVCVImageBatchVarShapeBufferPitch varShapePitch;
} NVCVImageBatchBuffer;

/** Stores information about image batch characteristics and content. */
typedef struct NVCVImageBatchDataRec
{
    /** Image format. */
    NVCVImageFormat format;

    /** Number of images in the image batch */
    int32_t numImages;

    /** Type of image batch buffer.
     *  It defines which member of the \ref NVCVImageBatchBuffer tagged union that
     *  must be used to access the image batch buffer contents. */
    NVCVImageBatchBufferType bufferType;

    /** Stores the image batch contents. */
    NVCVImageBatchBuffer buffer;
} NVCVImageBatchData;

#endif // NVCV_IMAGEBATCHDATA_H
