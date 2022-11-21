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

#include <fmt/PixelType.hpp>
#include <nvcv/PixelType.h>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/fmt/TLS.hpp>
#include <util/Assert.h>
#include <util/String.hpp>

#include <cstring>

namespace priv = nv::cv::priv;
namespace util = nv::cv::util;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakePixelType,
                (NVCVPixelType * outPixelType, NVCVDataKind dataKind, NVCVPacking packing))
{
    return priv::ProtectCall(
        [&]
        {
            if (outPixelType == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output pixel type cannot be NULL");
            }

            *outPixelType = priv::PixelType{dataKind, packing}.value();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetPacking, (NVCVPixelType type, NVCVPacking *outPacking))
{
    return priv::ProtectCall(
        [&]
        {
            if (outPacking == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output packing cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outPacking = ptype.packing();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetBitsPerPixel, (NVCVPixelType type, int32_t *outBPP))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBPP == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to bits per pixel output cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outBPP = ptype.bpp();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetBitsPerChannel, (NVCVPixelType type, int32_t *outBits))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBits == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to bits per channel output cannot be NULL");
            }

            priv::PixelType        ptype{type};
            std::array<int32_t, 4> tmp = ptype.bpc();
            static_assert(sizeof(tmp) == 4 * sizeof(*outBits));
            memcpy(outBits, &tmp, sizeof(tmp)); // no UB!
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetDataKind, (NVCVPixelType type, NVCVDataKind *outDataKind))
{
    return priv::ProtectCall(
        [&]
        {
            if (outDataKind == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type output cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outDataKind = ptype.dataKind();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetNumChannels, (NVCVPixelType type, int32_t *outNumChannels))
{
    return priv::ProtectCall(
        [&]
        {
            if (outNumChannels == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to number of channels output cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outNumChannels = ptype.numChannels();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetChannelType,
                (NVCVPixelType type, int32_t channel, NVCVPixelType *outChannelType))
{
    return priv::ProtectCall(
        [&]
        {
            if (outChannelType == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to channel type output cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outChannelType = ptype.channelType(channel).value();
        });
}

NVCV_DEFINE_API(0, 0, const char *, nvcvPixelTypeGetName, (NVCVPixelType type))
{
    priv::FormatTLS &tls = priv::GetFormatTLS(); // noexcept

    char         *buffer  = tls.bufPixelTypeName;
    constexpr int bufSize = sizeof(tls.bufPixelTypeName);

    try
    {
        util::BufferOStream(buffer, bufSize) << priv::PixelType{type};

        using namespace std::literals;

        util::ReplaceAllInline(buffer, bufSize, "NVCV_DATA_KIND_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_PACKING_"sv, ""sv);
    }
    catch (std::exception &e)
    {
        strncpy(buffer, e.what(), bufSize - 1);
        buffer[bufSize - 1] = '\0';
    }
    catch (...)
    {
        strncpy(buffer, "Unexpected error retrieving NVCVPixelType string representation", bufSize - 1);
        buffer[bufSize - 1] = '\0';
    }

    return buffer;
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetStrideBytes, (NVCVPixelType type, int32_t *pixStride))
{
    return priv::ProtectCall(
        [&]
        {
            if (pixStride == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to pixel stride output cannot be NULL");
            }

            priv::PixelType ptype{type};
            *pixStride = ptype.strideBytes();
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvPixelTypeGetAlignment, (NVCVPixelType type, int32_t *outAlignment))
{
    return priv::ProtectCall(
        [&]
        {
            if (outAlignment == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to alignment cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outAlignment = ptype.alignment();
        });
}
