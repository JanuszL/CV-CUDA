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
 * @file OpLaplacian.h
 *
 * @brief Defines types and functions to handle the Laplacian operation.
 * @defgroup NVCV_C_ALGORITHM_LAPLACIAN Laplacian
 * @{
 */

#ifndef NVCV_OP_LAPLACIAN_H
#define NVCV_OP_LAPLACIAN_H

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

/** Constructs an instance of the Laplacian.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopLaplacianCreate(NVCVOperatorHandle *handle);

/** Executes the Laplacian operation on the given cuda stream.  This operation does not wait for completion.
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
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
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
 *      16bit Signed   | No
 *      32bit Unsigned | No
 *      32bit Signed   | No
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
 * @param [in] ksize Aperture size used to compute the second-derivative filters, it can be 1 or 3.
 *
 * @param [in] scale Scale factor for the Laplacian values (use 1 for no scale).
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopLaplacianSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                NVCVTensorHandle out, int32_t ksize, float scale,
                                                NVCVBorderType borderMode);

/**
 * Executes the Laplacian operation on a batch of images.
 *
 * @param[in] in Input image batch.
 * @param[out] out Output image batch.
 * @param[in] ksize Aperture size to compute second-derivative filters, either 1 or 3 per image, as a 1D Tensor of int.
 *                  + Must be of pixel type NVCV_PIXEL_TYPE_S32
 * @param[in] scale Scale factor Laplacian values as a 1D Tensor of float.
 *                  + Must be of pixel type NVCV_PIXEL_TYPE_F32
 * @param[in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopLaplacianVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                        NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                        NVCVTensorHandle ksize, NVCVTensorHandle scale,
                                                        NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_LAPLACIAN_H */
