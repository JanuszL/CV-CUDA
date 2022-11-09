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

#ifndef NVCV_PRIV_COLORSPEC_HPP
#define NVCV_PRIV_COLORSPEC_HPP

#include <nvcv/ColorSpec.h>

#include <memory> // for std::pair

namespace nv::cv::priv {

struct ChromaLoc
{
    NVCVChromaLocation horiz, vert;
};

class ColorSpec
{
public:
    constexpr ColorSpec(NVCVColorSpec cspec)
        : m_cspec{cspec}
    {
    }

    ColorSpec(NVCVColorSpace cspace, NVCVYCbCrEncoding encoding, NVCVColorTransferFunction xferfunc,
              NVCVColorRange range, const ChromaLoc &loc) noexcept;

    operator NVCVColorSpec() const noexcept;

    ChromaLoc chromaLoc() const noexcept;
    ColorSpec chromaLoc(const ChromaLoc &newLoc) const;

    NVCVColorSpace colorSpace() const noexcept;
    ColorSpec      colorSpace(NVCVColorSpace newColorSpace) const;

    NVCVYCbCrEncoding YCbCrEncoding() const noexcept;
    ColorSpec         YCbCrEncoding(NVCVYCbCrEncoding newEncoding) const;

    NVCVColorTransferFunction xferFunc() const noexcept;
    ColorSpec                 xferFunc(NVCVColorTransferFunction newXferFunc) const;

    NVCVColorRange colorRange() const noexcept;
    ColorSpec      colorRange(NVCVColorRange range) const;

    NVCVWhitePoint whitePoint() const noexcept;

private:
    NVCVColorSpec m_cspec;
};

std::ostream &operator<<(std::ostream &out, ColorSpec cspec);

NVCVChromaSubsampling MakeNVCVChromaSubsampling(int samplesHoriz, int samplesVert);
std::pair<int, int>   GetChromaSamples(NVCVChromaSubsampling css);

bool NeedsColorspec(NVCVColorModel cmodel);

const char *GetName(NVCVColorModel colorModel);
const char *GetName(NVCVColorSpec colorSpec);
const char *GetName(NVCVChromaSubsampling chromaSub);
const char *GetName(NVCVColorTransferFunction xferFunc);
const char *GetName(NVCVYCbCrEncoding cstd);
const char *GetName(NVCVColorRange range);
const char *GetName(NVCVWhitePoint whitePoint);
const char *GetName(NVCVColorSpace color_space);
const char *GetName(NVCVChromaLocation loc);
const char *GetName(NVCVRawPattern raw);

} // namespace nv::cv::priv

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

// To be used inside gdb, as sometimes it has problems resolving the correct
// overload based on the parameter type.
std::string StrNVCVColorSpec(NVCVColorSpec cspec);

#endif // NVCV_PRIV_COLORSPEC_HPP
