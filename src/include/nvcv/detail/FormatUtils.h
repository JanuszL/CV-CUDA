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

#ifndef NVCV_DETAIL_FORMATUTILS_H
#define NVCV_DETAIL_FORMATUTILS_H

// Internal implementation of pixel formats/types.
// Not to be used directly.

#include <stdint.h>

// Utilities ================================

#define NVCV_DETAIL_SET_BITFIELD(value, offset, size) (((uint64_t)(value) & ((1ULL << (size)) - 1)) << (offset))
#define NVCV_DETAIL_GET_BITFIELD(value, offset, size) (((uint64_t)(value) >> (offset)) & ((1ULL << (size)) - 1))

// MAKE_COLOR_SPEC =======================================

#define NVCV_DETAIL_MAKE_COLOR_SPEC(CSpace, Encoding, XferFunc, Range, LocHoriz, LocVert)   \
    (NVCV_DETAIL_SET_BITFIELD((CSpace), 0, 3) | NVCV_DETAIL_SET_BITFIELD(XferFunc, 3, 4)    \
     | NVCV_DETAIL_SET_BITFIELD(Encoding, 7, 3) | NVCV_DETAIL_SET_BITFIELD(LocHoriz, 10, 2) \
     | NVCV_DETAIL_SET_BITFIELD(LocVert, 12, 2) | NVCV_DETAIL_SET_BITFIELD(Range, 14, 1))

#define NVCV_DETAIL_MAKE_CSPC(CSpace, Encoding, XferFunc, Range, LocHoriz, LocVert)                                    \
    NVCV_DETAIL_MAKE_COLOR_SPEC(NVCV_COLOR_##CSpace, NVCV_YCbCr_##Encoding, NVCV_COLOR_##XferFunc, NVCV_COLOR_##Range, \
                                NVCV_CHROMA_##LocHoriz, NVCV_CHROMA_##LocVert)

#endif /* NVCV_DETAIL_FORMATUTILS_H */
