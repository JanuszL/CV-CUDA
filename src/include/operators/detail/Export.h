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
