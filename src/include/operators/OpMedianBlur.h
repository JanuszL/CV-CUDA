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
 * @file OpMedianBlur.h
 *
 * @brief Defines types and functions to handle the median blur operation.
 * @defgroup NVCV_C_ALGORITHM_MEDIAN_BLUR MedianBlur
 * @{
 */

#ifndef NVCV_OP_MEDIAN_BLUR_H
#define NVCV_OP_MEDIAN_BLUR_H

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

/** Constructs and an instance of the median blur operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @param [in] maxVarShapeBatchSize maximum batch size for var shape operator
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopMedianBlurCreate(NVCVOperatorHandle *handle, const int maxVarShapeBatchSize);

/** Executes the median blur operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1, 3, 4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | No
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | Yes
 *       Data Type     | Yes
 *       Number        | Yes
 *       Channels      | Yes
 *       Width         | Yes
 *       Height        | Yes
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor / image batch.
 *
 * @param [out] out output tensor / image batch.
 *
 * @param [in] kernelWidth width of the kernel.
 *
 * @param [in] kernelHeight height of the kernel.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
/** @{ */
NVCV_OP_PUBLIC NVCVStatus nvcvopMedianBlurSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out, const int kernelWidth, const int KernelHeight);

NVCV_OP_PUBLIC NVCVStatus nvcvopMedianBlurVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                         NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                         NVCVTensorHandle ksize);
/** @} */

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_MEDIAN_BLUR_H */
