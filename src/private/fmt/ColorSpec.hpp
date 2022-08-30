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

} // namespace nv::cv::priv

// To be used inside gdb, as sometimes it has problems resolving the correct
// overload based on the parameter type.
std::string StrNVCVColorSpec(NVCVColorSpec cspec);

#endif // NVCV_PRIV_COLORSPEC_HPP
