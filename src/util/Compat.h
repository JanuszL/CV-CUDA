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

#ifndef NVCV_UTIL_COMPAT_H
#define NVCV_UTIL_COMPAT_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C"
{
#endif

ssize_t Compat_getrandom(void *buffer, size_t length, unsigned int flags);
int     Compat_getentropy(void *buffer, size_t length);

#ifdef __cplusplus
}
#endif

#endif
