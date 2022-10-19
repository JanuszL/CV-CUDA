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

#ifndef NVCV_UTIL_RECT_H
#define NVCV_UTIL_RECT_H

#ifdef __cplusplus
extern "C"
{
#endif

struct NVCVRectI
{
    int32_t x;      //!< x coordinate of the top-left corner
    int32_t y;      //!< y coordinate of the top-left corner
    int32_t width;  //!< width of the rectangle
    int32_t height; //!< height of the rectangle
};

#ifdef __cplusplus
}
#endif

#endif // NVCV_UTIL_RECT_H
