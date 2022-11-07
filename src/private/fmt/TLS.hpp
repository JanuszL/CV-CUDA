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

#ifndef NVCV_PRIV_TLS_HPP
#define NVCV_PRIV_TLS_HPP

#include <exception>

namespace nv::cv::priv {

struct FormatTLS
{
    char bufColorSpecName[1024];
    char bufPixelTypeName[1024];
    char bufImageFormatName[1024];
};

FormatTLS &GetFormatTLS() noexcept;

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TLS_HPP
