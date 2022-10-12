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

#include "TensorData.hpp"

#include "Exception.hpp"
#include "TensorLayout.hpp"

namespace nv::cv::priv {

void ValidateImageFormatForTensor(ImageFormat fmt)
{
    if (fmt.memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED,
                        "Tensor image batch of block-linear format images is not currently supported.");
    }

    if (fmt.css() != NVCV_CSS_444)
    {
        throw Exception(NVCV_ERROR_NOT_IMPLEMENTED)
            << "Batch image format must not have subsampled planes, but it is: " << fmt;
    }

    if (fmt.numPlanes() != 1 && fmt.numPlanes() != fmt.numChannels())
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format cannot be semi-planar, but it is: " << fmt;
    }

    for (int p = 1; p < fmt.numPlanes(); ++p)
    {
        if (fmt.planePacking(p) != fmt.planePacking(0))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Format's planes must all have the same packing, but they don't: " << fmt;
        }
    }
}

NVCVTensorLayout GetTensorLayoutFor(ImageFormat fmt, int nbatches)
{
    (void)nbatches;

    int nplanes = fmt.numPlanes();

    if (nplanes == 1)
    {
        return NVCV_TENSOR_NHWC;
    }
    else if (nplanes == fmt.numChannels())
    {
        return NVCV_TENSOR_NCHW;
    }
    else
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format cannot be semi-planar, but it is: " << fmt;
    }
}

// Layout, format and shape are already validated
static void DoFillTensorData(NVCVTensorData &data, ImageFormat format, NVCVTensorLayout layout, const int32_t *shape,
                             void *mem, const int64_t *pitchBytes)
{
    data.format     = format.value();
    data.bufferType = NVCV_TENSOR_BUFFER_PITCH_DEVICE;

    NVCVTensorBufferPitch &buffer = data.buffer.pitch;

    buffer.layout = layout;
    buffer.mem    = mem;

    int ndims = GetNDims(buffer.layout);

    memcpy(buffer.shape, shape, sizeof(*shape) * ndims);

    if (pitchBytes != nullptr)
    {
        memcpy(buffer.pitchBytes, pitchBytes, sizeof(*pitchBytes) * ndims);
    }
    else
    {
        // defines a packed tensor

        switch (buffer.layout)
        {
        case NVCV_TENSOR_NCHW:
            buffer.pitchBytes[ndims - 1] = format.planePixelStrideBytes(0);
            break;
        case NVCV_TENSOR_NHWC:
            buffer.pitchBytes[ndims - 1] = format.planePixelStrideBytes(0) / format.numChannels();
            break;
        default:
            NVCV_ASSERT(!"Invalid tensor layout");
        }

        for (int dim = ndims - 2; dim >= 0; --dim)
        {
            buffer.pitchBytes[dim] = buffer.pitchBytes[dim + 1] * buffer.shape[dim + 1];
        }
    }
}

void FillTensorData(NVCVTensorData &data, ImageFormat format, const int32_t *shape, void *mem,
                    const int64_t *pitchBytes)
{
    if (shape == nullptr)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to shape must not be NULL");
    }

    ValidateImageFormatForTensor(format);

    NVCVTensorLayout layout = GetTensorLayoutFor(format, shape[0]);

    DoFillTensorData(data, format, layout, shape, mem, pitchBytes);
}

void FillTensorData(NVCVTensorData &data, ImageFormat format, const priv::DimsNCHW &dims, void *mem,
                    const int64_t *pitchBytes)
{
    ValidateImageFormatForTensor(format);

    NVCVTensorLayout layout = GetTensorLayoutFor(format, dims.n);

    Shape shape;

    switch (layout)
    {
    case NVCV_TENSOR_NCHW:
        shape = {dims.n, dims.c, dims.h, dims.w};
        break;
    case NVCV_TENSOR_NHWC:
        shape = {dims.n, dims.h, dims.w, dims.c};
        break;
    default:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid tensor layout: " << layout;
    }

    DoFillTensorData(data, format, layout, &shape[0], mem, pitchBytes);
}

void FillTensorData(NVCVTensorData &data, ImageFormat format, int32_t numImages, const Size2D &imgSize, void *mem,
                    const int64_t *pitchBytes)
{
    ValidateImageFormatForTensor(format);

    NVCVTensorLayout layout = GetTensorLayoutFor(format, numImages);

    Shape shape;

    switch (layout)
    {
    case NVCV_TENSOR_NCHW:
        shape = {numImages, format.numChannels(), imgSize.h, imgSize.w};
        break;
    case NVCV_TENSOR_NHWC:
        shape = {numImages, imgSize.h, imgSize.w, format.numChannels()};
        break;
    default:
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Invalid tensor layout: " << layout;
    }

    DoFillTensorData(data, format, layout, &shape[0], mem, pitchBytes);
}

} // namespace nv::cv::priv
