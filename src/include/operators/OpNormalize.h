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
 * @file OpNormalize.h
 *
 * @brief Defines types and functions to handle the normalize operation.
 * @defgroup NVCV_C_ALGORITHM_NORMALIZE Normalize
 * @{
 */

#ifndef NVCV_OP_NORMALIZE_H
#define NVCV_OP_NORMALIZE_H

#include "Operator.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the normalize operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopNormalizeCreate(NVCVOperatorHandle *handle);

/** Executes the normalize operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in intput tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] scale_is_stddev
 *
 * @param [in] global_scale
 *
 * @param [in] shift
 *
 * @param [in] epsilon
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopNormalizeSubmit(NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in,
                                                NVCVTensorHandle out, const bool scale_is_stddev,
                                                const float global_scale, const float shift, const float epsilon);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_NORMALIZE_H */