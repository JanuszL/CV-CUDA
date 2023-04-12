/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file OpBndBox.h
 *
 * @brief Defines types and functions to handle the BndBox operation.
 * @defgroup NVCV_C_ALGORITHM__BND_BOX BndBox
 * @{
 */

#ifndef CVCUDA__BND_BOX_H
#define CVCUDA__BND_BOX_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Rect.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the BndBox operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaBndBoxCreate(NVCVOperatorHandle *handle);

/** Executes the BndBox operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [TODO]
 *       Channels:       [TODO]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | TODO
 *       8bit  Signed   | TODO
 *       16bit Unsigned | TODO
 *       16bit Signed   | TODO
 *       32bit Unsigned | TODO
 *       32bit Signed   | TODO
 *       32bit Float    | TODO
 *       64bit Float    | TODO
 *
 *  Output:
 *       Data Layout:    [TODO]
 *       Channels:       [TODO]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | TODO
 *       8bit  Signed   | TODO
 *       16bit Unsigned | TODO
 *       16bit Signed   | TODO
 *       32bit Unsigned | TODO
 *       32bit Signed   | TODO
 *       32bit Float    | TODO
 *       64bit Float    | TODO
 *
 *  Input/Output dependency
 *
 *       Property      |  Input == Output
 *      -------------- | -------------
 *       Data Layout   | TODO
 *       Data Type     | TODO
 *       Number        | TODO
 *       Channels      | TODO
 *       Width         | TODO
 *       Height        | TODO
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in intput tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] bbox bounding box rectangle in reference to the input tensor.
 *
 * @param [in] thickness border thickness for bounding box rectangle.
 *
 * @param [in] borderColor border color for bounding box rectangle.
 *
 * @param [in] fillColor fill color for bounding box rectangle.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
CVCUDA_PUBLIC NVCVStatus cvcudaBndBoxSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                            NVCVTensorHandle out, const NVCVRectI bbox, int thickness, uchar4 borderColor, uchar4 fillColor, bool enableMSAA);

#ifdef __cplusplus
}
#endif

#endif /* CVCUDA__BND_BOX_H */
