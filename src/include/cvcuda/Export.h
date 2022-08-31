/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
* @brief Export : Defines all macros used
*
*/
#ifndef NV_CVCUDA_EXPORT_H
#define NV_CVCUDA_EXPORT_H

#if defined _WIN32 || defined __CYGWIN__
#    ifdef CVCUDA_EXPORTING
#        define CVCUDA_PUBLIC __declspec(dllexport)
#    elif defined(CVCUDA_STATIC)
#        define CVCUDA_PUBLIC
#    else
#        define CVCUDA_PUBLIC __declspec(dllimport)
#    endif
#else
#    if __GNUC__ >= 4
#        define CVCUDA_PUBLIC __attribute__((visibility("default")))
#    else
#        define CVCUDA_PUBLIC
#    endif
#endif

#endif /* NV_CVCUDA_EXPORT_H */
