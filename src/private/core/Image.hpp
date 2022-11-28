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

#ifndef NVCV_PRIV_IMAGE_HPP
#define NVCV_PRIV_IMAGE_HPP

#include "IImage.hpp"

namespace nv::cv::priv {

class Image final : public CoreObjectBase<IImage>
{
public:
    explicit Image(NVCVImageRequirements reqs, IAllocator &alloc);
    ~Image();

    static NVCVImageRequirements CalcRequirements(Size2D size, ImageFormat fmt);

    Size2D        size() const override;
    ImageFormat   format() const override;
    IAllocator   &alloc() const override;
    NVCVTypeImage type() const override;

    void exportData(NVCVImageData &data) const override;

private:
    IAllocator           &m_alloc;
    NVCVImageRequirements m_reqs;
    void                 *m_buffer;
};

class ImageWrapData final : public CoreObjectBase<IImage>
{
public:
    explicit ImageWrapData(const NVCVImageData &data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup);

    ~ImageWrapData();

    Size2D        size() const override;
    ImageFormat   format() const override;
    IAllocator   &alloc() const override;
    NVCVTypeImage type() const override;

    void exportData(NVCVImageData &data) const override;

private:
    NVCVImageData m_data;

    NVCVImageDataCleanupFunc m_cleanup;
    void                    *m_ctxCleanup;

    void doCleanup() noexcept;

    void doValidateData(const NVCVImageData &data) const;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_IMAGE_HPP
