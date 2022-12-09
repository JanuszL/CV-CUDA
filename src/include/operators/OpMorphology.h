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
 * @file OpMorphology.h
 *
 * @brief Defines types and functions to handle the morphology operation.
 * @defgroup NVCV_C_ALGORITHM_MORPHOLOGY Morphology
 * @{
 */

#ifndef NVCV_OP_MORPHOLOGY_H
#define NVCV_OP_MORPHOLOGY_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Size.hpp>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the Morphology operator.
 *
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 * @param [in] maxVarShapeBatchSize maximum batch size for var shape operator, can be 0 if VarShape is not used.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopMorphologyCreate(NVCVOperatorHandle *handle, const int32_t maxVarShapeBatchSize);

/**
 * Executes the morphology operation of Dilates/Erodes on images
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
 * @param [in] morphType Type of operation to performs Erode/Dilate. \ref NVCVMorphologyType.
 *
 * @param [in] maskWidth Width of the mask to use (set heigh/width to -1 for default of 3,3).
 *
 * @param [in] maskHeight Height of the mask to use (set heigh/width to -1 for default of 3,3).
 *
 * @param [in] anchorX X-offset of the kernel to use (set anchorX/anchorY to -1 for center of kernel).
 *
 * @param [in] anchorY Y-offset of the kernel to use (set anchorX/anchorY to -1 for center of kernel).
 *
 * @param [in] iteration  Number of times to execute the operation, typically set to 1. Setting to higher than 1 is equivelent
 *                       of increasing the kernel mask by (mask_width - 1, mask_height -1) for every iteration.

 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \ref NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopMorphologySubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                 NVCVTensorHandle out, NVCVMorphologyType morphType, int32_t maskWidth,
                                                 int32_t maskHeight, int32_t anchorX, int32_t anchorY,
                                                 int32_t iteration, const NVCVBorderType borderMode);

/**
 * Executes the morphology operation of Dilates/Erodes on images, using an array of variable shape images.
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
 *      Width         | Yes (per image in/out pair)
 *      Height        | Yes (per image in/out pair)
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input variable shape tensor.
 *
 * @param [out] out Output variable shape tensor.
 *
 * @param [in] morphType Type of operation to perform (Erode/Dilate). \ref NVCVMorphologyType.
 *
 * @param [in] masks  1D Tensor of NVCV_PIXEL_TYPE_2S32 mask W/H pairs, where the 1st pair is for image 0, second for image 1, etc.
 *                    Setting values to -1,-1 will create a default 3,3 mask.
 *                    (Note after the operation the tensor values may be modified by kernel)
 *
 * @param [in] anchors 1D Tensor of NVCV_PIXEL_TYPE_2S32 X/Y pairs, where the 1st pair is for image 0, second for image 1, etc
 *                      Setting values to -1,-1 will anchor the kernel at the center.
 *                      (Note after the operation the tensor values may be modified by kernel)
 *
 * @param [in] iteration Number of times to execute the operation, typically set to 1. Setting to higher than 1 is equivelent
 *                       of increasing the kernel mask by (mask_width - 1, mask_height -1) for every iteration.

 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \ref NVCVBorderType.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopMorphologyVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                         NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                         NVCVMorphologyType morphType, NVCVTensorHandle masks,
                                                         NVCVTensorHandle anchors, int32_t iteration,
                                                         const NVCVBorderType borderMode);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_MORPHOLOGY */
