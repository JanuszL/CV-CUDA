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

#include "AllocInfo.hpp"
#include "IImage.hpp"

namespace nv::cv::priv {

class Image final : public IImage
{
public:
    explicit Image(Size2D size, ImageFormat fmt, IAllocator &alloc);
    ~Image();

    Size2D        size() const override;
    ImageFormat   format() const override;
    IAllocator   &alloc() const override;
    NVCVTypeImage type() const override;

    void exportData(NVCVImageData &data) const override;

private:
    IAllocator &m_alloc;
    AllocInfo2D m_allocInfo;
    Size2D      m_size;
    ImageFormat m_format;
    void       *m_buffer;

    Version doGetVersion() const override;
};

class ImageWrapData final : public IImageWrapData
{
public:
    explicit ImageWrapData(IAllocator &alloc);

    explicit ImageWrapData(const NVCVImageData &data, IAllocator &alloc);

    Size2D        size() const override;
    ImageFormat   format() const override;
    IAllocator   &alloc() const override;
    NVCVTypeImage type() const override;

    void exportData(NVCVImageData &data) const override;

    void setData(const NVCVImageData &data) override;
    void resetData() override;

private:
    NVCVImageData m_data;
    IAllocator   &m_alloc;

    Version doGetVersion() const override;

    void doValidateData(const NVCVImageData &data) const;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_IMAGE_HPP
