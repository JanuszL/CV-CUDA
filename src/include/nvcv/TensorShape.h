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

#ifndef NVCV_TENSORSHAPE_H
#define NVCV_TENSORSHAPE_H

#include "TensorLayout.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Permute the input shape with given layout to a different layout.
 *
 * @param[in] srcLayout The layout of the source shape.
 *
 * @param[in] srcShape The shape to be permuted.
 *                     + Must not be NULL.
 *                     + Number of dimensions must be equal to dimensions in @p srcLayout
 *
 * @param[in] dstLayout The layout of the destination shape.
 *
 * @param[out] dstShape Where the permutation will be written to.
 *                      + Must not be NULL.
 *                      + Number of dimensions must be equal to dimensions in @p dstLayout
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some parameter is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvTensorShapePermute(NVCVTensorLayout srcLayout, const int64_t *srcShape,
                                              NVCVTensorLayout dstLayout, int64_t *dstShape);

#ifdef __cplusplus
}
#endif

#endif // NVCV_TENSORSHAPE_H
