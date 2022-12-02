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
 * @file OpWarpPerspective.h
 *
 * @brief Defines types and functions to handle the WarpPerspective operation.
 * @defgroup NVCV_C_ALGORITHM_WARP_PERSPECTIVE WarpPerspective
 * @{
 */

#ifndef NVCV_OP_WARP_PERSPECTIVE_H
#define NVCV_OP_WARP_PERSPECTIVE_H

#include "Operator.h"
#include "Types.h"
#include "detail/Export.h"

#include <cuda_runtime.h>
#include <nvcv/ImageBatch.h>
#include <nvcv/Status.h>
#include <nvcv/Tensor.h>
#include <nvcv/cuda/math/LinAlg.hpp>

// @brief storage for perspective transform matrix (row major)
typedef float NVCVPerspectiveTransform[9];

#ifdef __cplusplus
extern "C"
{
#endif

/** Constructs and an instance of the WarpPerspective operator.
 *
 * @param [out] handle Where the image instance handle will be written to.
 *                     + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Handle is null.
 * @retval #NVCV_ERROR_OUT_OF_MEMORY    Not enough memory to create the operator.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopWarpPerspectiveCreate(NVCVOperatorHandle *handle, const int32_t maxVarShapeBatchSize);

/** Executes the WarpPerspective operation on the given cuda stream. This operation does not
 *  wait for completion.
 *
 *  Applies an perspective transformation to an image.
 *  outputs(x,y) = saturate_cast<out_type>(input(transform(x,y)))
 *  where transform() is the linear transformation operator (matrix)
 *
 *  Limitations:
 *
 *  Input:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1,3,4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
 *       32bit Unsigned | No
 *       32bit Signed   | No
 *       32bit Float    | Yes
 *       64bit Float    | No
 *
 *  Output:
 *       Data Layout:    [kNHWC, kHWC]
 *       Channels:       [1,3,4]
 *
 *       Data Type      | Allowed
 *       -------------- | -------------
 *       8bit  Unsigned | Yes
 *       8bit  Signed   | No
 *       16bit Unsigned | Yes
 *       16bit Signed   | Yes
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
 *       Width         | No
 *       Height        | No
 *
 * @param [in] handle Handle to the operator.
 *                    + Must not be NULL.
 * @param [in] stream Handle to a valid CUDA stream.
 *
 * @param [in] in input tensor.
 *
 * @param [out] out output tensor.
 *
 * @param [in] transMatrix 3x3 perspective transformation matrix.
 *
 * @param [in] flags Combination of interpolation methods(NVCV_INTERP_NEAREST, NVCV_INTERP_LINEAR or NVCV_INTERP_CUBIC)
                     and the optional flag NVCV_WARP_INVERSE_MAP, that sets trans_matrix as the inverse transformation.
 *
 * @param [in] borderMode pixel extrapolation method (NVCV_BORDER_CONSTANT or NVCV_BORDER_REPLICATE).
 *
 * @param [in] borderValue used in case of a constant border.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside valid range.
 * @retval #NVCV_ERROR_INTERNAL         Internal error in the operator, invalid types passed in.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_OP_PUBLIC NVCVStatus nvcvopWarpPerspectiveSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                      NVCVTensorHandle in, NVCVTensorHandle out,
                                                      const NVCVPerspectiveTransform transMatrix, const int32_t flags,
                                                      const NVCVBorderType borderMode, const float4 borderValue);

NVCV_OP_PUBLIC NVCVStatus nvcvopWarpPerspectiveVarShapeSubmit(NVCVOperatorHandle handle, cudaStream_t stream,
                                                              NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                                                              NVCVTensorHandle transMatrix, const int32_t flags,
                                                              const NVCVBorderType borderMode,
                                                              const float4         borderValue);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_WARP_PERSPECTIVE_H */
