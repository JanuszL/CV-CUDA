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

#include <fmt/DataLayout.hpp>
#include <nvcv/DataLayout.h>
#include <nvcv/DataLayout.hpp>
#include <nvcv/Status.h>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <util/Assert.h>
#include <util/SymbolVersioning.hpp>

#include <cstring>

namespace priv = nv::cv::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakeSwizzle,
                (NVCVSwizzle * outSwizzle, NVCVChannel x, NVCVChannel y, NVCVChannel z, NVCVChannel w))
{
    return priv::ProtectCall(
        [&]
        {
            if (outSwizzle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output swizzle cannot be NULL");
            }

            *outSwizzle = priv::MakeNVCVSwizzle(x, y, z, w);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvSwizzleGetChannels, (NVCVSwizzle swizzle, NVCVChannel *outChannels))
{
    return priv::ProtectCall(
        [&]
        {
            if (outChannels == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output channel array cannot be NULL");
            }

            std::array<NVCVChannel, 4> tmp = priv::GetChannels(swizzle);
            static_assert(sizeof(tmp) == sizeof(*outChannels) * 4);
            memcpy(outChannels, &tmp, sizeof(tmp)); // no UB!
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvSwizzleGetNumChannels, (NVCVSwizzle swizzle, int32_t *outNumChannels))
{
    return priv::ProtectCall(
        [&]
        {
            if (outNumChannels == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to number of channels output cannot be NULL");
            }

            *outNumChannels = priv::GetNumChannels(swizzle);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvMakePacking, (NVCVPacking * outPacking, const NVCVPackingParams *params))
{
    return priv::ProtectCall(
        [&]
        {
            if (params == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to input packing parameters must not be NULL");
            }

            if (outPacking == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output packing must not be NULL");
            }

            if (std::optional<NVCVPacking> packing = priv::MakeNVCVPacking(*params))
            {
                *outPacking = *packing;
            }
            else
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Parameters don't correspond to any supported packing");
            }
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPackingGetParams, (NVCVPacking packing, NVCVPackingParams *outParams))
{
    return priv::ProtectCall(
        [&]
        {
            if (outParams == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to output parameters must not be NULL");
            }
            *outParams = priv::GetPackingParams(packing);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPackingGetNumComponents, (NVCVPacking packing, int32_t *outNumComponents))
{
    return priv::ProtectCall(
        [&]
        {
            if (outNumComponents == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to number of components output must not be NULL");
            }

            *outNumComponents = priv::GetNumComponents(packing);
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPackingGetBitsPerComponent, (NVCVPacking packing, int32_t *outBits))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBits == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to number of bits per component output must not be NULL");
            }

            std::array<int32_t, 4> tmp = priv::GetBitsPerComponent(packing);
            static_assert(sizeof(tmp) == sizeof(*outBits) * 4);
            memcpy(outBits, &tmp, sizeof(tmp)); // No UB!
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvPackingGetBitsPerPixel, (NVCVPacking packing, int32_t *outBPP))
{
    return priv::ProtectCall(
        [&]
        {
            if (outBPP == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT,
                                      "Pointer to number of bits per pixel output must not be NULL");
            }

            *outBPP = priv::GetBitsPerPixel(packing);
        });
}
