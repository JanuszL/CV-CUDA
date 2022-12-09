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
 * @file OpGaussian.h
 *
 * @brief Defines types and functions to handle the Gaussian operation.
 * @defgroup NVCV_C_ALGORITHM_GAUSSIAN Gaussian
 * @{
 */

#ifndef NVCV_OP_GAUSSIAN_H
#define NVCV_OP_GAUSSIAN_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the Gaussian.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxKernelWidth The maximum kernel width that will be used by the operator.
 *                            + Positive value.
 * @param [in] maxKernelHeight The maximum kernel height that will be used by the operator.
 *                             + Positive value.
 * @param [in] maxVarShapeBatchSize The maximum batch size that will be used by the var-shape operator.
 *                                  + Positive value.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopGaussianCreate(NVCVOperatorHandle *handle, int32_t maxKernelWidth,
                                               int32_t maxKernelHeight, int32_t maxVarShapeBatchSize);

/** Executes the Gaussian operation on the given cuda stream.  This operation does not wait for completion.
 *
 * Limitations:
 *
 * Input:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Output:
 *      Data Layout:    [kNHWC, kHWC]
 *      Channels:       [1, 3, 4]
 *
 *      Data Type      | Allowed
 *      -------------- | -------------
 *      8bit  Unsigned | Yes
 *      8bit  Signed   | No
 *      16bit Unsigned | Yes
 *      16bit Signed   | Yes
 *      32bit Unsigned | No
 *      32bit Signed   | Yes
 *      32bit Float    | Yes
 *      64bit Float    | No
 *
 * Input/Output dependency
 *
 *      Property      |  Input == Output
 *     -------------- | -------------
 *      Data Layout   | Yes
 *      Data Type     | Yes
 *      Number        | Yes
 *      Channels      | Yes
 *      Width         | Yes
 *      Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input tensor.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] kernelWidth Gaussian kernel width.
 *
 * @param [in] kernelHeight Gaussian kernel height.
 *
 * @param [in] sigmaX Gaussian kernel standard deviation in X direction.
 *
 * @param [in] sigmaY Gaussian kernel standard deviation in Y direction.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopGaussianSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                               NVCVTensorHandle out, int32_t kernelWidth, int32_t kernelHeight,
                                               double sigmaX, double sigmaY, NVCVBorderType borderMode);

/**
 * Executes the Gaussian operation on a batch of images.
 *
 * @param[in] in Input image batch.
 * @param[out] out Output image batch.
 * @param[in] kernelSize Gaussian kernel size as a Tensor of int2.
 *                       + Must be of pixel type NVCV_PIXEL_TYPE_2S32
 * @param[in] sigma Gaussian sigma as a Tensor of double2.
 *                  + Must be of pixel type NVCV_PIXEL_TYPE_2F64
 * @param[in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopGaussianVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                       NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                       NVCVTensorHandle kernelSize, NVCVTensorHandle sigma,
                                                       NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_GAUSSIAN_H */
