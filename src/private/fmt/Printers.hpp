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

#ifndef NVCV_PRIV_FORMAT_PRINTERS_HPP
#define NVCV_PRIV_FORMAT_PRINTERS_HPP

#include <nvcv/ColorSpec.h>
#include <nvcv/DataLayout.h>

#include <iosfwd>

// ColorSpec
std::ostream &operator<<(std::ostream &out, NVCVColorModel colorModel);
std::ostream &operator<<(std::ostream &out, NVCVColorSpec colorSpec);
std::ostream &operator<<(std::ostream &out, NVCVChromaSubsampling chromaSub);
std::ostream &operator<<(std::ostream &out, NVCVColorTransferFunction xferFunc);
std::ostream &operator<<(std::ostream &out, NVCVYCbCrEncoding cstd);
std::ostream &operator<<(std::ostream &out, NVCVColorRange range);
std::ostream &operator<<(std::ostream &out, NVCVWhitePoint whitePoint);
std::ostream &operator<<(std::ostream &out, NVCVColorSpace color_space);
std::ostream &operator<<(std::ostream &out, NVCVChromaLocation loc);
std::ostream &operator<<(std::ostream &out, NVCVRawPattern raw);

// DataType
std::ostream &operator<<(std::ostream &out, NVCVDataType dataType);
std::ostream &operator<<(std::ostream &out, const NVCVPacking &packing);
std::ostream &operator<<(std::ostream &out, NVCVMemLayout memLayout);
std::ostream &operator<<(std::ostream &out, NVCVChannel swizzleChannel);
std::ostream &operator<<(std::ostream &out, NVCVSwizzle swizzle);
std::ostream &operator<<(std::ostream &out, NVCVByteOrder byteOrder);

#endif // NVCV_PRIV_FORMAT_PRINTERS_HPP
