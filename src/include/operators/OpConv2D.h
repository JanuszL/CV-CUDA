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
 * @file OpConv2D.h
 *
 * @brief Defines types and functions to handle the 2D Convolution operation.
 * @defgroup NVCV_C_ALGORITHM_CONV2D 2D Convolution
 * @{
 */

#ifndef NVCV_OP_CONV2D_H
#define NVCV_OP_CONV2D_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs an instance of the Conv2D.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopConv2DCreate(NVCVOperatorHandle *handle);

/** Executes the Conv2D operation on the given cuda stream.  This operation does not wait for completion.
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
 * @param [in] kernel Convolution kernels (one for each batch image) to be used.  Each image width and height
 * correspond to the kernel width and height.
 *                    + Must be of pixel type NVCV_PIXEL_TYPE_F32
 *
 * @param [in] kernelAnchor 1D Tensor with the anchor of each kernel (one for each batch image).  The anchor (x, y)
 * indicates the relative position of a filtered point within the kernel.  (-1, -1) means that the anchor is at the
 * kernel center.
 *                          + Must be of pixel type NVCV_PIXEL_TYPE_2S32
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside, cf. \p NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopConv2DVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                     NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                     NVCVImageBatchHandle kernel, NVCVTensorHandle kernelAnchor,
                                                     NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_CONV2D_H */
