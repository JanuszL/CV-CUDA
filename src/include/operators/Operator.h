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
 * @file Operator.h
 *
 * @brief Defines types and functions to handle color specs.
 */

#ifndef NVCV_OP_OPERATOR_H
#define NVCV_OP_OPERATOR_H

#include "detail/Export.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct NVCVOperator *NVCVOperatorHandle;

NVCV_OP_PUBLIC void nvcvOperatorDestroy(NVCVOperatorHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_OP_OPERATOR_H */
