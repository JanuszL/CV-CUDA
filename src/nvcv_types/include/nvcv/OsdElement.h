/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVCV_OSD_ELEMENT_H
#define NVCV_OSD_ELEMENT_H

#include <nvcv/Rect.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} NVCVColor;

typedef struct
{
    NVCVRectI       rect;           //!< Rectangle of the bounding box, \ref NVCVRectI.
    int32_t         thickness;      //!< Border thickness of bounding box.
    NVCVColor       borderColor;    //!< Border color of bounding box.
    NVCVColor       fillColor;      //!< Filled color of bounding box.
} NVCVBndBoxI;

typedef struct
{
    NVCVBndBoxI*    boxes;          //!< Bounding box rectangle, \ref NVCVBndBoxI.
    int32_t         box_num;        //!< Bounding box num.
} NVCVBndBoxesI;

typedef struct
{
    NVCVRectI       rect;           //!< Rectangle of the blur box, \ref NVCVRectI.
    int32_t         kernelSize;     //!< Kernel sizes of mean filter, refer to cv::blur().
} NVCVBlurBoxI;

typedef struct
{
    NVCVBlurBoxI*   boxes;          //!< Blurring box rectangle, \ref NVCVBlurBoxI.
    int32_t         box_num;        //!< Blurring box num.
} NVCVBlurBoxesI;

#ifdef __cplusplus
}
#endif

#endif // NVCV_OSD_ELEMENT_H
