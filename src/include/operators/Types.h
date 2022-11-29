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
 * @file Types.h
 *
 * @brief Defines types to be used by operators.
 * @defgroup NVCV_C_TYPES Types
 * @{
 */

#ifndef NVCV_TYPES_H
#define NVCV_TYPES_H

#include "detail/Export.h"

#ifdef __cplusplus
extern "C"
{
#endif

// @brief Flag to choose the interpolation method to be used
typedef enum
{
    NVCV_INTERP_NEAREST,
    NVCV_INTERP_LINEAR,
    NVCV_INTERP_CUBIC,
    NVCV_INTERP_AREA,
} NVCVInterpolationType;

// @brief Flag to choose the border mode to be used
typedef enum
{
    NVCV_BORDER_CONSTANT   = 0,
    NVCV_BORDER_REPLICATE  = 1,
    NVCV_BORDER_REFLECT    = 2,
    NVCV_BORDER_WRAP       = 3,
    NVCV_BORDER_REFLECT101 = 4,
} NVCVBorderType;

typedef enum
{
    NVCV_ERODE  = 0,
    NVCV_DILATE = 1,
} NVCVMorphologyType;

#ifdef __cplusplus
}
#endif

#endif /* NVCV_TYPES_H */
