/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * Functions and structures for handling NVCV operator library version.
 */

#ifndef NVCV_OP_VERSION_H
#define NVCV_OP_VERSION_H

#include "detail/Export.h"

#include <nvcv/operators/VersionDef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/** Retrieves the library's version number.
 * The number is represented as a integer. It may differ from \ref NVCV_OP_VERSION if
 * header doesn't correspond to NVCV operator binary. This can be used by user's program
 * to handle semantic differences between library versions.
 */
NVCV_OP_PUBLIC uint32_t nvcvopGetVersion(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif // NVCV_OP_VERSION_H
