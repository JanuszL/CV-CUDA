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
 * @file OpAverageBlur.h
 *
 * @brief Defines types and functions to handle the AverageBlur operation.
 * @defgroup NVCV_C_ALGORITHM_AVERAGEBLUR Average Blur
 * @{
 */

#ifndef NVCV_OP_AVERAGEBLUR_H
#define NVCV_OP_AVERAGEBLUR_H

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

/** Constructs an instance of the AverageBlur.
 *
 * @param [out] handle Where the operator instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxKernelWidth The maximum kernel width that will be used by the operator.
 *                            + Positive value.
 * @param [in] maxKernelHeight The maximum kernel height that will be used by the operator.
 *                            + Positive value.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopAverageBlurCreate(NVCVOperatorHandle *handle, int maxKernelWidth, int maxKernelHeight);

/** Executes the AverageBlur operation on the given cuda stream.  This operation does not wait for completion.
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
 * @param [in] kernelWidth AverageBlur kernel width.
 *
 * @param [in] kernelHeight AverageBlur kernel height.
 *
 * @param [in] kernelAnchorX Kernel anchor in X direction.  Indicates the relative position of a filtered point
 * within the kernel.  Use (-1, -1) to indicate kernel center.
 *
 * @param [in] kernelAnchorY Kernel anchor in Y direction.  Indicates the relative position of a filtered point
 * within the kernel.  Use (-1, -1) to indicate kernel center.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopAverageBlurSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                  NVCVTensorHandle out, int kernelWidth, int kernelHeight,
                                                  int kernelAnchorX, int kernelAnchorY, NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_AVERAGEBLUR_H */