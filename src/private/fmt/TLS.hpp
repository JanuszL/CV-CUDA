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

#ifndef NVCV_PRIV_TLS_HPP
#define NVCV_PRIV_TLS_HPP

#include <exception>

namespace nv::cv::priv {

struct FormatTLS
{
    char bufColorSpecName[1024];
    char bufColorModelName[128];
    char bufChromaLocationName[128];
    char bufRawPatternName[128];
    char bufColorSpaceName[128];
    char bufColorTransferFunctionName[128];
    char bufColorRangeName[128];
    char bufWhitePointName[128];
    char bufYCbCrEncodingName[128];
    char bufChromaSubsamplingName[128];

    char bufDataTypeName[128];
    char bufMemLayoutName[128];
    char bufChannelName[128];
    char bufSwizzleName[128];
    char bufByteOrderName[128];
    char bufPackingName[128];

    char bufPixelTypeName[1024];
    char bufImageFormatName[1024];
};

FormatTLS &GetFormatTLS() noexcept;

} // namespace nv::cv::priv

#endif // NVCV_PRIV_TLS_HPP
