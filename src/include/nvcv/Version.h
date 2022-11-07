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
 * @file Version.h
 *
 * Functions and structures for handling NVCV library version.
 */

#ifndef NVCV_VERSION_H
#define NVCV_VERSION_H

#include "Export.h"

#include <nvcv/VersionDef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Retrieves the library's version number.
 * The number is represented as a integer. It may differ from \ref NVCV_VERSION if
 * header doesn't correspond to NVCV binary. This can be used by user's program
 * to handle semantic differences between library versions.
 */
NVCV_PUBLIC uint32_t nvcvGetVersion(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // NVCV_VERSION_H
