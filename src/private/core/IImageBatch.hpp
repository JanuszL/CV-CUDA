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

#ifndef NVCV_PRIV_IIMAGEBATCH_HPP
#define NVCV_PRIV_IIMAGEBATCH_HPP

#include "ICoreObject.hpp"

#include <fmt/ImageFormat.hpp>
#include <nvcv/ImageBatch.h>

namespace nv::cv::priv {

class IAllocator;

class IImageBatch : public ICoreObjectHandle<IImageBatch, NVCVImageBatchHandle>
{
public:
    virtual int32_t     capacity() const = 0;
    virtual ImageFormat format() const   = 0;
    virtual int32_t     size() const     = 0;

    virtual NVCVTypeImageBatch type() const = 0;

    virtual IAllocator &alloc() const = 0;

    virtual void exportData(CUstream stream, NVCVImageBatchData &data) const = 0;
};

class IImageBatchVarShape : public IImageBatch
{
public:
    virtual void pushImages(const NVCVImageHandle *images, int32_t numImages) = 0;
    virtual void pushImages(NVCVPushImageFunc cbPushImage, void *ctxCallback) = 0;

    virtual void popImages(int32_t numImages) = 0;

    virtual void clear() = 0;

    virtual void getImages(int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages) const = 0;
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_IIMAGEBATCH_HPP
