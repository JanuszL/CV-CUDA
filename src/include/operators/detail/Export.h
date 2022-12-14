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

/**
* @file Export.h
*
* @brief Export : Defines all macros used for NVCV OP
*
*/

#ifndef NVCVOP_EXPORT_H
#define NVCVOP_EXPORT_H

#if defined _WIN32 || defined __CYGWIN__
#    ifdef NVCV_EXPORTING
#        define NVCV_OP_PUBLIC __declspec(dllexport)
#    elif defined(NVCV_STATIC)
#        define NVCV_OP_PUBLIC
#    else
#        define NVCV_OP_PUBLIC __declspec(dllimport)
#    endif
#else
#    if __GNUC__ >= 4
#        define NVCV_OP_PUBLIC __attribute__((visibility("default")))
#    else
#        define NVCV_OP_PUBLIC
#    endif
#endif

#endif /* NVCVOP_EXPORT_H */
