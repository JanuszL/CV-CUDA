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
 * @file OpPadAndStack.h
 *
 * @brief Defines types and functions to handle the pad and stack operation.
 * @defgroup NVCV_C_ALGORITHM_PADANDSTACK Pad and stack
 * @{
 */

#ifndef NVCV_OP_PADANDSTACK_H
#define NVCV_OP_PADANDSTACK_H

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

/** Constructs an instance of the pad and stack operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopPadAndStackCreate(NVCVOperatorHandle *handle);

/** Executes the pad and stack operation on the given cuda stream. This operation does not wait for completion.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in Input image batch.
 *
 * @param [out] out Output tensor.
 *
 * @param [in] hleft Left tensor to store amount of left padding per batch input image.
 *                   This tensor is a vector of integers, where the elements are stored in the width of the tensor
 *                   and the other dimensions are 1, i.e. n=h=c=1.
 *
 * @param [in] htop Top tensor to store amount of top padding per batch input image.
 *                  This tensor is a vector of integers, where the elements are stored in the width of the tensor
 *                  and the other dimensions are 1, i.e. n=h=c=1.
 *
 * @param [in] borderMode Border mode to be used when accessing elements outside input image, cf. \p NVCVBorderType.
 *
 * @param [in] borderValue Border value to be used for constant border mode \p NVCV_BORDER_CONSTANT.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopPadAndStackSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                  NVCVImageBatchHandle in, NVCVTensorHandle out, NVCVTensorHandle hleft,
                                                  NVCVTensorHandle htop, const NVCVBorderType borderMode,
                                                  const float borderValue);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_PADANDSTACK_H */
