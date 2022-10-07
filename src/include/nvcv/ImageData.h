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

#ifndef NVCV_IMAGEDATA_H
#define NVCV_IMAGEDATA_H

#include "ImageFormat.h"
#include "PixelType.h"
#include "detail/CudaFwd.h"

#include <stdint.h>

typedef struct NVCVImagePlanePitchRec
{
    /** Width of this plane in pixels.
     *  + It must be >= 1. */
    int32_t width;

    /** Height of this plane in pixels.
     *  + It must be >= 1. */
    int32_t height;

    /** Difference in bytes of beginning of one row and the beginning of the previous.
         This is used to address every row (and ultimately every pixel) in the plane.
         @code
            T *pix_addr = (T *)((uint8_t *)data + pitchBytes*height)+width;
         @endcode
         where T is the C type related to pixelType.

         + It must be at least `(width * bits-per-pixel + 7)/8`.
    */
    int32_t pitchBytes;

    /** Pointer to the beginning of the first row of this plane.
        This points to the actual plane contents. */
    void *buffer;
} NVCVImagePlanePitch;

/** Maximum number of data planes an image can have. */
#define NVCV_MAX_PLANE_COUNT (6)

/** Stores the image plane contents. */
typedef struct NVCVImageBufferPitchRec
{
    /** Number of planes.
     *  + Must be >= 1. */
    int32_t numPlanes;

    /** Data of all image planes in pitch-linear layout.
     *  + Only the first \ref numPlanes elements must have valid data. */
    NVCVImagePlanePitch planes[NVCV_MAX_PLANE_COUNT];
} NVCVImageBufferPitch;

typedef struct NVCVImageBufferCudaArrayRec
{
    /** Number of planes.
     *  + Must be >= 1. */
    int32_t numPlanes;

    /** Data of all image planes in pitch-linear layout.
     *  + Only the first \ref numPlanes elements must have valid data. */
    cudaArray_t planes[NVCV_MAX_PLANE_COUNT];
} NVCVImageBufferCudaArray;

/** Represents how the image data is stored. */
typedef enum
{
    /** Invalid buffer type.
     *  This is commonly used to inform that no buffer type was selected. */
    NVCV_IMAGE_BUFFER_NONE = 0,

    /** GPU-accessible with planes in pitch-linear layout. */
    NVCV_IMAGE_BUFFER_DEVICE_PITCH,

    /** Buffer stored in a cudaArray_t.
     * Please consult <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays">cudaArray_t</a>
     * for more information. */
    NVCV_IMAGE_BUFFER_CUDA_ARRAY,
} NVCVImageBufferType;

/** Represents the available methods to access image contents.
 * The correct method depends on \ref NVCVImageData::bufferType. */
typedef union NVCVImageBufferRec
{
    /** Image stored in pitch-linear layout.
     * To be used when \ref NVCVImageData::bufferType is:
     * - \ref NVCV_IMAGE_BUFFER_HOST_PITCH
     * - \ref NVCV_IMAGE_BUFFER_HOST_PINNED_PITCH
     * - \ref NVCV_IMAGE_BUFFER_DEVICE_PITCH
     */
    NVCVImageBufferPitch pitch;

    /** Image stored in a `cudaArray_t`.
     * To be used when \ref NVCVImageData::bufferType is:
     * - \ref NVCV_IMAGE_BUFFER_CUDA_ARRAY
     */
    NVCVImageBufferCudaArray cudaarray;
} NVCVImageBuffer;

// Forward declaration
typedef struct NVCVImageDataRec NVCVImageData;

/** Stores information about image characteristics and content. */
typedef struct NVCVImageDataRec
{
    /** Image format. */
    NVCVImageFormat format;

    /** Type of image buffer.
     *  It defines which member of the \ref NVCVImageBuffer tagged union that
     *  must be used to access the image contents. */
    NVCVImageBufferType bufferType;

    /** Stores the image contents. */
    NVCVImageBuffer buffer;
} NVCVImageData;

#endif // NVCV_IMAGEDATA_H
