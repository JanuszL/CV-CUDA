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

#ifdef __cplusplus
extern "C"
{
#endif

/** Tensor memory layout.
 * Describes how to interpret tensor shape's elements. */
typedef enum
{
    NVCV_TENSOR_NCHW,
    NVCV_TENSOR_NHWC,
} NVCVTensorLayout;

#define NVCV_TENSOR_MAX_NDIM (4)

/** Stores the tensor plane contents. */
typedef struct NVCVTensorBufferPitchRec
{
    NVCVTensorLayout layout;

    int32_t shape[NVCV_TENSOR_MAX_NDIM];
    int64_t pitchBytes[NVCV_TENSOR_MAX_NDIM];

    /** Pointer to memory buffer with tensor contents.
     * Pixel with type T is addressed by:
     * pixAttr = (uint8_t *)mem + shape[0]*pitchBytes[0] + ... + shape[ndim-1]*pitchBytes[ndim-1];
     */
    void *mem;
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
    /** Image format. */
    NVCVImageFormat format;

    /** Type of image batch buffer.
     *  It defines which member of the \ref NVCVTensorBuffer tagged union that
     *  must be used to access the image batch buffer contents. */
    NVCVTensorBufferType bufferType;

    /** Stores the image batch contents. */
    NVCVTensorBuffer buffer;
} NVCVTensorData;

/**
 * Fills a tensor data with pitch-linear buffer specified by its NCHW dimensions and layout.
 *
 * Tensor layout will be NCHW or NHWC depending whether the image format is
 * is packed or planar.
 *
 * @param[out] data Where the tensor data will be written to.
 * @param[in] format Tensor image format
 * @param[in] nbatch,channels,height,width Tensor dimensions
 * @param[in] mem Pointer to memory buffer with tensor contents.
 * @param[in] pitchBytes Pitch of each dimension.
 *                       If NULL, will define a fully packed tensor.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorDataPitchDeviceFillDimsNCHW(NVCVTensorData *data, NVCVImageFormat format,
                                                             int32_t nbatch, int32_t channels, int32_t height,
                                                             int32_t width, void *mem, const int64_t *pitchBytes);

/**
 * Fills a tensor data with pitch-linear buffer specified by image attributes.
 *
 * Tensor layout will be NCHW or NHWC depending whether the image format is
 * is packed or planar.
 *
 * @param[out] data Where the tensor data will be written to.
 * @param[in] format Tensor image format
 * @param[in] numImages Number of images in the tensor.
 * @param[in] imgWidth,imgHeight Dimension of each image.
 * @param[in] mem Pointer to memory buffer with tensor contents.
 * @param[in] pitchBytes Pitch of each dimension.
 *                       If NULL, will define a fully packed tensor.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorDataPitchDeviceFillForImages(NVCVTensorData *data, NVCVImageFormat format,
                                                              int32_t numImages, int32_t imgWidth, int32_t imgHeight,
                                                              void *mem, const int64_t *pitchBytes);

/**
 * Retrieve the number of dimensions of a tensor layout.
 *
 * @param[in] layout Tensor layout to be queried.
 *
 * @param[out] ndim Number of dimensions of the tensor layout.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorLayoutGetNumDim(NVCVTensorLayout layout, int32_t *ndim);

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORDATA_H
