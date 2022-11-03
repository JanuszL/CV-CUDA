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

#include "ImageBatchVarShape.hpp"

#include "IAllocator.hpp"
#include "IImage.hpp"
#include "Requirements.hpp"

#include <cuda_runtime.h>
#include <fmt/PixelType.hpp>
#include <util/CheckError.hpp>
#include <util/Math.hpp>

#include <cmath>
#include <numeric>

namespace nv::cv::priv {

// ImageBatchVarShape implementation -------------------------------------------

NVCVImageBatchVarShapeRequirements ImageBatchVarShape::CalcRequirements(int32_t capacity, ImageFormat fmt)
{
    NVCVImageBatchVarShapeRequirements reqs;
    reqs.capacity = capacity;
    reqs.format   = fmt.value();
    reqs.mem      = {};

    reqs.alignBytes = alignof(NVCVImagePlanePitch);
    reqs.alignBytes = std::lcm(alignof(NVCVImageHandle), reqs.alignBytes);

    reqs.alignBytes = util::RoundUpNextPowerOfTwo(reqs.alignBytes);

    if (reqs.alignBytes > NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Alignment requirement of %d is larger than the maximum allowed %ld", reqs.alignBytes,
                        NVCV_MAX_MEM_REQUIREMENTS_BLOCK_SIZE);
    }

    AddBuffer(reqs.mem.deviceMem, capacity * sizeof(NVCVImagePlanePitch), reqs.alignBytes);
    AddBuffer(reqs.mem.hostMem, capacity * sizeof(NVCVImagePlanePitch), reqs.alignBytes);
    AddBuffer(reqs.mem.hostMem, capacity * sizeof(NVCVImageHandle), reqs.alignBytes);

    return reqs;
}

ImageBatchVarShape::ImageBatchVarShape(NVCVImageBatchVarShapeRequirements reqs, IAllocator &alloc)
    : m_alloc{alloc}
    , m_reqs{std::move(reqs)}
    , m_dirtyStartingFromIndex(0)
    , m_numImages(0)
{
    ImageFormat fmt{m_reqs.format};

    if (fmt.memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED,
                        "Image batch of block-linear format images is not currently supported.");
    }

    m_evPostFence     = nullptr;
    m_devPlanesBuffer = m_hostPlanesBuffer = nullptr;
    m_imgHandleBuffer                      = nullptr;

    int64_t bufPlanesSize  = m_reqs.capacity * sizeof(NVCVImagePlanePitch) * fmt.numPlanes();
    int64_t imgHandlesSize = m_reqs.capacity * sizeof(NVCVImageHandle);

    try
    {
        m_devPlanesBuffer
            = reinterpret_cast<NVCVImagePlanePitch *>(m_alloc.allocDeviceMem(bufPlanesSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_devPlanesBuffer != nullptr);

        m_hostPlanesBuffer
            = reinterpret_cast<NVCVImagePlanePitch *>(m_alloc.allocHostMem(bufPlanesSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_devPlanesBuffer != nullptr);

        m_imgHandleBuffer
            = reinterpret_cast<NVCVImageHandle *>(m_alloc.allocHostMem(imgHandlesSize, m_reqs.alignBytes));
        NVCV_ASSERT(m_imgHandleBuffer != nullptr);

        NVCV_CHECK_THROW(cudaEventCreateWithFlags(&m_evPostFence, cudaEventDisableTiming));
    }
    catch (...)
    {
        if (m_evPostFence)
        {
            NVCV_CHECK_LOG(cudaEventDestroy(m_evPostFence));
        }

        m_alloc.freeDeviceMem(m_devPlanesBuffer, bufPlanesSize, m_reqs.alignBytes);
        m_alloc.freeHostMem(m_hostPlanesBuffer, bufPlanesSize, m_reqs.alignBytes);
        m_alloc.freeHostMem(m_imgHandleBuffer, imgHandlesSize, m_reqs.alignBytes);
        throw;
    }
}

ImageBatchVarShape::~ImageBatchVarShape()
{
    NVCV_CHECK_LOG(cudaEventSynchronize(m_evPostFence));

    int64_t bufPlanesSize  = m_reqs.capacity * sizeof(NVCVImagePlanePitch) * this->format().numPlanes();
    int64_t imgHandlesSize = m_reqs.capacity * sizeof(NVCVImageHandle);

    m_alloc.freeDeviceMem(m_devPlanesBuffer, bufPlanesSize, m_reqs.alignBytes);
    m_alloc.freeHostMem(m_hostPlanesBuffer, bufPlanesSize, m_reqs.alignBytes);
    m_alloc.freeHostMem(m_imgHandleBuffer, imgHandlesSize, m_reqs.alignBytes);

    NVCV_CHECK_LOG(cudaEventDestroy(m_evPostFence));
}

NVCVTypeImageBatch ImageBatchVarShape::type() const
{
    return NVCV_TYPE_IMAGEBATCH_VARSHAPE;
}

Version ImageBatchVarShape::doGetVersion() const
{
    return CURRENT_VERSION;
}

int32_t ImageBatchVarShape::capacity() const
{
    return m_reqs.capacity;
}

int32_t ImageBatchVarShape::numImages() const
{
    return m_numImages;
}

ImageFormat ImageBatchVarShape::format() const
{
    return ImageFormat{m_reqs.format};
}

IAllocator &ImageBatchVarShape::alloc() const
{
    return m_alloc;
}

void ImageBatchVarShape::exportData(CUstream stream, NVCVImageBatchData &data) const
{
    ImageFormat fmt{m_reqs.format};

    NVCV_ASSERT(fmt.memLayout() == NVCV_MEM_LAYOUT_PL);

    data.format     = m_reqs.format;
    data.numImages  = m_numImages;
    data.bufferType = NVCV_IMAGE_BATCH_VARSHAPE_BUFFER_PITCH_DEVICE;

    NVCVImageBatchVarShapeBufferPitch &buf = data.buffer.varShapePitch;
    buf.imgPlanes                          = m_devPlanesBuffer;

    NVCV_ASSERT(m_dirtyStartingFromIndex <= m_numImages);

    if (m_dirtyStartingFromIndex < m_numImages)
    {
        int numPlanes = this->format().numPlanes();

        NVCV_CHECK_THROW(
            cudaMemcpyAsync(m_devPlanesBuffer + m_dirtyStartingFromIndex * numPlanes,
                            m_hostPlanesBuffer + m_dirtyStartingFromIndex * numPlanes,
                            (m_numImages - m_dirtyStartingFromIndex) * sizeof(*m_devPlanesBuffer) * numPlanes,
                            cudaMemcpyHostToDevice, stream));

        // Signal that we finished reading from m_hostBuffer
        NVCV_CHECK_THROW(cudaEventRecord(m_evPostFence, stream));

        // up to m_numImages, we're all good
        m_dirtyStartingFromIndex = m_numImages;
    }
}

void ImageBatchVarShape::pushImages(const NVCVImageHandle *images, int32_t numImages)
{
    if (images == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Images buffer cannot be NULL");
    }

    if (numImages + m_numImages > m_reqs.capacity)
    {
        throw Exception(NVCV_ERROR_OVERFLOW,
                        "Adding %d images to image batch would make its size %d exceed its capacity %d", numImages,
                        numImages + m_numImages, m_reqs.capacity);
    }

    // Wait till m_hostBuffer is free to be written to (all pending reads are finished).
    NVCV_CHECK_THROW(cudaEventSynchronize(m_evPostFence));

    int oldNumImages = m_numImages;

    try
    {
        for (int i = 0; i < numImages; ++i)
        {
            doPushImage(images[i]);
        }
    }
    catch (...)
    {
        m_numImages = oldNumImages;
        throw;
    }
}

void ImageBatchVarShape::pushImages(NVCVPushImageFunc cbPushImage, void *ctxCallback)
{
    if (cbPushImage == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT,
                        "Callback function that adds images to the image batch cannot be NULL");
    }

    // Wait till m_hostBuffer is free to be written to (all pending reads are finished).
    NVCV_CHECK_THROW(cudaEventSynchronize(m_evPostFence));

    int oldNumImages = m_numImages;

    try
    {
        while (NVCVImageHandle imgHandle = cbPushImage(ctxCallback))
        {
            if (m_numImages == m_reqs.capacity)
            {
                throw Exception(NVCV_ERROR_OVERFLOW,
                                "Adding one more image to image batch would make its size exceed its capacity %d",
                                m_reqs.capacity);
            }

            doPushImage(imgHandle);
        }
    }
    catch (...)
    {
        m_numImages = oldNumImages;
        throw;
    }
}

void ImageBatchVarShape::doPushImage(NVCVImageHandle imgHandle)
{
    NVCV_ASSERT(m_numImages < m_reqs.capacity);

    auto &img = ToStaticRef<IImage>(imgHandle);

    if (img.format() != this->format())
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Format of image to be added, " << img.format()
                                                     << ", is different from image batch format " << this->format();
    }

    NVCVImageData imgData;
    img.exportData(imgData);

    if (imgData.bufferType != NVCV_IMAGE_BUFFER_PITCH_DEVICE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Data buffer of image to be added isn't gpu-accessible";
    }

    int numPlanes = imgData.buffer.pitch.numPlanes;

    std::copy(imgData.buffer.pitch.planes, imgData.buffer.pitch.planes + numPlanes,
              m_hostPlanesBuffer + m_numImages * numPlanes);

    m_imgHandleBuffer[m_numImages] = imgHandle;

    ++m_numImages;
}

void ImageBatchVarShape::popImages(int32_t numImages)
{
    if (m_numImages - numImages < 0)
    {
        throw Exception(NVCV_ERROR_UNDERFLOW,
                        "Cannot remove more images, %d, than the number of images, %d, in the image batch", numImages,
                        m_numImages);
    }

    m_numImages -= numImages;

    if (m_dirtyStartingFromIndex > m_numImages)
    {
        m_dirtyStartingFromIndex = m_numImages;
    }
}

void ImageBatchVarShape::getImages(int32_t begIndex, NVCVImageHandle *outImages, int32_t numImages) const
{
    if (begIndex + numImages > m_numImages)
    {
        throw Exception(NVCV_ERROR_OVERFLOW, "Cannot get images past end of image batch");
    }

    std::copy(m_imgHandleBuffer + begIndex, m_imgHandleBuffer + begIndex + numImages, outImages);
}

void ImageBatchVarShape::clear()
{
    m_numImages              = 0;
    m_dirtyStartingFromIndex = 0;
}

} // namespace nv::cv::priv
