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
 * @file PixelType.h
 *
 * @brief Defines types and functions to handle pixel types.
 */

#ifndef NVCV_PIXEL_TYPE_H
#define NVCV_PIXEL_TYPE_H

#include "ColorSpec.h"
#include "DataLayout.h"
#include "detail/FormatUtils.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Pre-defined pixel types.
 * Pixel types defines the geometry of pixels in a image plane without taking into account what the value represents.
 * For example, a \ref NVCV_IMAGE_FORMAT_NV12 is composed of 2 planes, each one with the following pixel types:
 * + \ref NVCV_PIXEL_TYPE_U8 representing pixels as 8-bit unsigned values.
 * + \ref NVCV_PIXEL_TYPE_2U8 representing pixels as two interleaved 32-bit floating-point values.
 */
typedef uint64_t NVCVPixelType;

/* clang-format off */

/** Denotes a special image format that doesn't represent any particular type (like void). */
#define NVCV_PIXEL_TYPE_NONE ((NVCVPixelType)0)

/** One channel of unsigned 8-bit value. */
#define NVCV_PIXEL_TYPE_U8   NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X8)
/** Two interleaved channels of unsigned 8-bit values. */
#define NVCV_PIXEL_TYPE_2U8  NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X8_Y8)
/** Three interleaved channels of unsigned 8-bit values. */
#define NVCV_PIXEL_TYPE_3U8  NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X8_Y8_Z8)
/** Four interleaved channels of unsigned 8-bit values. */
#define NVCV_PIXEL_TYPE_4U8  NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X8_Y8_Z8_W8)

/** One channel of signed 8-bit value. */
#define NVCV_PIXEL_TYPE_S8   NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X8)
/** Two interleaved channels of signed 8-bit values. */
#define NVCV_PIXEL_TYPE_2S8  NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X8_Y8)
/** Three interleaved channels of signed 8-bit values. */
#define NVCV_PIXEL_TYPE_3S8  NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X8_Y8_Z8)
/** Four interleaved channels of signed 8-bit values. */
#define NVCV_PIXEL_TYPE_4S8  NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X8_Y8_Z8_W8)

/** One channel of unsigned 16-bit value. */
#define NVCV_PIXEL_TYPE_U16  NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X16)
/** Two interleaved channels of unsigned 16-bit values. */
#define NVCV_PIXEL_TYPE_2U16 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X16_Y16)
/** Three interleaved channels of unsigned 16-bit values. */
#define NVCV_PIXEL_TYPE_3U16 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X16_Y16_Z16)
/** Four interleaved channels of unsigned 16-bit values. */
#define NVCV_PIXEL_TYPE_4U16 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X16_Y16_Z16_W16)

/** One channel of signed 16-bit value. */
#define NVCV_PIXEL_TYPE_S16  NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X16)
/** Two interleaved channels of signed 16-bit values. */
#define NVCV_PIXEL_TYPE_2S16 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X16_Y16)
/** Three interleaved channels of signed 16-bit values. */
#define NVCV_PIXEL_TYPE_3S16 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X16_Y16_Z16)
/** Four interleaved channels of signed 16-bit values. */
#define NVCV_PIXEL_TYPE_4S16 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X16_Y16_Z16_W16)

/** One channel of unsigned 32-bit value. */
#define NVCV_PIXEL_TYPE_U32  NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X32)
/** Two interleaved channels of unsigned 32-bit values. */
#define NVCV_PIXEL_TYPE_2U32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X32_Y32)
/** Three interleaved channels of unsigned 32-bit values. */
#define NVCV_PIXEL_TYPE_3U32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X32_Y32_Z32)
/** Four interleaved channels of unsigned 32-bit values. */
#define NVCV_PIXEL_TYPE_4U32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X32_Y32_Z32_W32)

/** One channel of signed 32-bit value. */
#define NVCV_PIXEL_TYPE_S32  NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X32)
/** Two interleaved channels of signed 32-bit values. */
#define NVCV_PIXEL_TYPE_2S32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X32_Y32)
/** Three interleaved channels of signed 32-bit values. */
#define NVCV_PIXEL_TYPE_3S32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X32_Y32_Z32)
/** Four interleaved channels of signed 32-bit values. */
#define NVCV_PIXEL_TYPE_4S32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X32_Y32_Z32_W32)

/** One channel of 32-bit IEEE 754 floating-point value. */
#define NVCV_PIXEL_TYPE_F32  NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X32)
/** Two interleaved channels of 32-bit IEEE 754 floating-point values. */
#define NVCV_PIXEL_TYPE_2F32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X32_Y32)
/** Three interleaved channels of 32-bit IEEE 754 floating-point values. */
#define NVCV_PIXEL_TYPE_3F32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X32_Y32_Z32)
/** Four interleaved channels of 32-bit IEEE 754 floating-point values. */
#define NVCV_PIXEL_TYPE_4F32 NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X32_Y32_Z32_W32)

/** One channel of unsigned 64-bit value. */
#define NVCV_PIXEL_TYPE_U64  NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X64)
/** Two interleaved channels of unsigned 64-bit values. */
#define NVCV_PIXEL_TYPE_2U64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X64_Y64)
/** Three interleaved channels of unsigned 64-bit values. */
#define NVCV_PIXEL_TYPE_3U64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X64_Y64_Z64)
/** Four interleaved channels of unsigned 64-bit values. */
#define NVCV_PIXEL_TYPE_4U64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, UNSIGNED, X64_Y64_Z64_W64)

/** One channel of signed 64-bit value. */
#define NVCV_PIXEL_TYPE_S64  NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X64)
/** Two interleaved channels of signed 64-bit values. */
#define NVCV_PIXEL_TYPE_2S64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X64_Y64)
/** Three interleaved channels of signed 64-bit values. */
#define NVCV_PIXEL_TYPE_3S64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X64_Y64_Z64)
/** Four interleaved channels of signed 64-bit values. */
#define NVCV_PIXEL_TYPE_4S64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, SIGNED, X64_Y64_Z64_W64)

/** One channel of 64-bit IEEE 754 floating-point value. */
#define NVCV_PIXEL_TYPE_F64  NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X64)
/** Two interleaved channels of 64-bit IEEE 754 floating-point values. */
#define NVCV_PIXEL_TYPE_2F64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X64_Y64)
/** Three interleaved channels of 64-bit IEEE 754 floating-point values. */
#define NVCV_PIXEL_TYPE_3F64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X64_Y64_Z64)
/** Four interleaved channels of 64-bit IEEE 754 floating-point values. */
#define NVCV_PIXEL_TYPE_4F64 NVCV_DETAIL_MAKE_PIX_TYPE(PL, FLOAT, X64_Y64_Z64_W64)

/* clang-format on */

/** Creates a user-defined pixel type constant.
 *
 * Example to create a block-linear format two interleaved 32-bit floating point channels:
 * \code{.c}
 *     NVCVPixelType type = NVCV_MAKE_PIXEL_TYPE(NVCV_MEM_LAYOUT_BL, NVCV_DATA_TYPE_FLOAT, NVCV_PACKING_X32_Y32);
 * \endcode
 *
 * @param[in] memLayout \ref NVCVMemLayout to be used.
 * @param[in] dataType  \ref NVCVDataType to be used.
 * @param[in] packing   Format packing used, which also defines the number of channels.
 *
 * @returns The user-defined pixel type.
 */
#ifdef DOXYGEN_SHOULD_SKIP_THIS
#    define NVCV_MAKE_PIXEL_TYPE(memLayout, dataType, packing)
#else
#    define NVCV_MAKE_PIXEL_TYPE (NVCVPixelType) NVCV_DETAIL_MAKE_PIXEL_TYPE
#endif

/** Creates a user-defined pixel type.
 * When the pre-defined pixel types aren't enough, user-defined formats can be created.
 *
 * @param[out] outPixType The user-defined pixel type.
 *                        + Cannot be NULL.
 *
 * @param[in] memLayout \ref NVCVMemLayout to be used.
 * @param[in] dataType \ref NVCVDataType to be used.
 * @param[in] packing Format packing used, which also defines the number of channels.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvMakePixelType(NVCVPixelType *outPixType, NVCVMemLayout memLayout, NVCVDataType dataType,
                                         NVCVPacking packing);

/** Get the packing of a pixel type.
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[out] outPacking The format's packing.
 *                 + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetPacking(NVCVPixelType type, NVCVPacking *outPacking);

/** Get the number of bits per pixel of a pixel type.
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[out] outBPP The number of bits per pixel.
 *                    + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetBitsPerPixel(NVCVPixelType type, int32_t *outBPP);

/** Get the number of bits per channel of a pixel type.
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[out] outBits Pointer to an int32_t array with 4 elements where output will be stored.
 *                     If it is NULL, the function is a no-op, i.e., buffer pointed by \p bits is unchanged.
 *                     + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetBitsPerChannel(NVCVPixelType type, int32_t *outBits);

/** Get the data type of a pixel type.
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[out] outDataType The data type of the pixel type.
 *                      + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetDataType(NVCVPixelType type, NVCVDataType *outDataType);

/** Get the memory layout of a pixel type.
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[out] outMemLayout The memory layout of the pixel type.
 *                          + Cannot be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetMemLayout(NVCVPixelType type, NVCVMemLayout *outMemLayout);

/** Set the memory layout of a pixel type.
 *
 * @param[inout] type Pixel type to have its memory layout set.
 *                    + Must not be NULL.
 *
 * @param[in] layout The memory layout to be set.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeSetMemLayout(NVCVPixelType *type, NVCVMemLayout layout);

/** Get the number of channels of a pixel type.
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[out] outNumChannels The number of channels of the pixel type.
 *                            + Must not be NULL.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetNumChannels(NVCVPixelType type, int32_t *outNumChannels);

/** Returns a string representation of the pixel type.
 *
 * @param[in] type Pixel type to be returned.
 *
 * @returns The string representation of the pixel type.
 *          Returned string is valid until next call of this function from the same calling thread.
 *          Returned pointer must not be freed.
 */
NVCV_PUBLIC const char *nvcvPixelTypeGetName(NVCVPixelType type);

/** Get the pixel type for a given channel index.
 *
 * It returns a single-channel pixel type that corresponds to the given channel
 * of the input pixel type.
 *
 * For instance: The channel #2 of \ref NVCV_PIXEL_TYPE_3U8 is \ref NVCV_PIXEL_TYPE_U8.
 *
 * + The requested channel must have a type whose packing is one of the following:
 *   - \ref NVCV_PACKING_X1
 *   - \ref NVCV_PACKING_X2
 *   - \ref NVCV_PACKING_X4
 *   - \ref NVCV_PACKING_X8
 *   - \ref NVCV_PACKING_X16
 *   - \ref NVCV_PACKING_X24
 *   - \ref NVCV_PACKING_X32
 *   - \ref NVCV_PACKING_X48
 *   - \ref NVCV_PACKING_X64
 *   - \ref NVCV_PACKING_X96
 *   - \ref NVCV_PACKING_X128
 *   - \ref NVCV_PACKING_X192
 *   - \ref NVCV_PACKING_X256
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[in] channel Channel whose pixel type is to be returned.
 *                 + Must be between 0 and the maximum number of channels in \p type.
 *
 * @param[out] outChannelType The pixel type of the given channel. The memory layout and data type are the same as \p type.
 *
 * @retval #NVCV_ERROR_INVALID_ARGUMENT Some argument is outside its valid range.
 * @retval #NVCV_SUCCESS                Operation executed successfully.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetChannelType(NVCVPixelType type, int32_t channel, NVCVPixelType *outChannelTpe);

/** Returns the stride/size in bytes of the pixel in memory.
 *
 * @param[in] type Pixel type to be queried.
 *
 * @param[out] pixStrideBytes The size in bytes of the pixel
 *                            + Must not be NULL.
 */
NVCV_PUBLIC NVCVStatus nvcvPixelTypeGetStrideBytes(NVCVPixelType type, int32_t *pixStrideBytes);

#ifdef __cplusplus
}
#endif

#endif /* NVCV_PIXEL_TYPE_H */
