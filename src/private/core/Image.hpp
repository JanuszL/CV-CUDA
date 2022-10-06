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

class Image final : public IImage
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

    Version doGetVersion() const override;
};

class ImageWrapData final : public IImageWrapData
{
public:
    explicit ImageWrapData(IAllocator &alloc);

    explicit ImageWrapData(const NVCVImageData &data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup,
                           IAllocator &alloc);

    ~ImageWrapData();

    Size2D        size() const override;
    ImageFormat   format() const override;
    IAllocator   &alloc() const override;
    NVCVTypeImage type() const override;

    void exportData(NVCVImageData &data) const override;

    void setData(const NVCVImageData *data) override;
    void setDataAndCleanup(const NVCVImageData *data, NVCVImageDataCleanupFunc cleanup, void *ctxCleanup) override;

private:
    NVCVImageData m_data;
    IAllocator   &m_alloc;

    NVCVImageDataCleanupFunc m_cleanup;
    void                    *m_ctxCleanup;

    Version doGetVersion() const override;

    void doCleanup() noexcept;

    void doValidateData(const NVCVImageData &data) const;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_IMAGE_HPP
