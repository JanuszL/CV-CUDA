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

#include <fmt/PixelType.hpp>
#include <fmt/Printers.hpp>
#include <nvcv/PixelType.h>
#include <nvcv/PixelType.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/core/TLS.hpp>
#include <util/Assert.h>
#include <util/String.hpp>

#include <cstring>
#ifdef __GNUC__
#    undef __DEPRECATED
#endif
#include <strstream>

namespace priv = nv::cv::priv;
namespace util = nv::cv::util;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakePixelType,
                (NVCVPixelType * outPixelType, NVCVMemLayout memLayout, NVCVDataType dataType, NVCVPacking packing))
{
    return priv::ProtectCall(
        [&]
        {
            if (outPixelType == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output pixel type cannot be NULL");
            }

            *outPixelType = priv::PixelType{memLayout, dataType, packing}.value();
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

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetDataType, (NVCVPixelType type, NVCVDataType *outDataType))
{
    return priv::ProtectCall(
        [&]
        {
            if (outDataType == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to data type output cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outDataType = ptype.dataType();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeGetMemLayout, (NVCVPixelType type, NVCVMemLayout *outMemLayout))
{
    return priv::ProtectCall(
        [&]
        {
            if (outMemLayout == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to memory layout output cannot be NULL");
            }

            priv::PixelType ptype{type};
            *outMemLayout = ptype.memLayout();
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPixelTypeSetMemLayout, (NVCVPixelType * type, NVCVMemLayout newLayout))
{
    return priv::ProtectCall(
        [&]
        {
            if (type == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to input pixel type cannot be NULL");
            }

            priv::PixelType ptype{*type};
            *type = ptype.memLayout(newLayout).value();
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
    priv::TLS &tls = priv::GetTLS(); // noexcept

    char         *buffer  = tls.bufPixelTypeName;
    constexpr int bufSize = sizeof(tls.bufPixelTypeName);

    try
    {
        std::strstreambuf sbuf(buffer, bufSize, buffer);
        std::ostream      ss(&sbuf);

        priv::PixelType ptype{type};

        // Must insert EOS to make 'str' a correctly delimited string
        ss << ptype << '\0' << std::flush;

        using namespace std::literals;

        util::ReplaceAllInline(buffer, bufSize, "NVCV_MEM_LAYOUT_"sv, ""sv);
        util::ReplaceAllInline(buffer, bufSize, "NVCV_DATA_TYPE_"sv, ""sv);
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
