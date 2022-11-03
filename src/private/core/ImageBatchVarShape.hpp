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

#ifndef NVCV_PRIV_IMAGEBATCHVARSHAPE_HPP
#define NVCV_PRIV_IMAGEBATCHVARSHAPE_HPP

#include "IImageBatch.hpp"

#include <cuda_runtime.h>

namespace nv::cv::priv {

class ImageBatchVarShape final : public IImageBatchVarShape
{
public:
    explicit ImageBatchVarShape(NVCVImageBatchVarShapeRequirements reqs, IAllocator &alloc);
    ~ImageBatchVarShape();

    static NVCVImageBatchVarShapeRequirements CalcRequirements(int32_t capacity, ImageFormat fmt);

    int32_t     capacity() const override;
    ImageFormat format() const override;
    int32_t     numImages() const override;

    NVCVTypeImageBatch type() const override;

    IAllocator &alloc() const override;

    void getImages(int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages) const override;

    void exportData(CUstream stream, NVCVImageBatchData &data) const override;

    void pushImages(const NVCVImageHandle *images, int32_t numImages) override;
    void pushImages(NVCVPushImageFunc cbPushImage, void *ctxCallback) override;
    void popImages(int32_t numImages) override;
    void clear() override;

private:
    IAllocator                        &m_alloc;
    NVCVImageBatchVarShapeRequirements m_reqs;

    mutable int32_t m_dirtyStartingFromIndex;

    int32_t              m_numImages;
    NVCVImagePlanePitch *m_hostPlanesBuffer;
    NVCVImagePlanePitch *m_devPlanesBuffer;
    NVCVImageHandle     *m_imgHandleBuffer;

    // TODO: must be retrieved from the resource allocator;
    cudaEvent_t m_evPostFence;

    Version doGetVersion() const override;

    // Assumes there's enough space for image.
    // Does not update dirty count
    void doPushImage(NVCVImageHandle imgHandle);
};

} // namespace nv::cv::priv

#endif // NVCV_PRIV_IMAGEBATCHVARSHAPE_HPP
