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

#include <fmt/PixelType.hpp>
#include <nvcv/TensorLayout.h>

namespace nv::cv::priv {

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

void FillTensorData(IImage &img, NVCVTensorData &tensorData)
{
    ImageFormat fmt = img.format();

    // Must do a lot of checks to see if image is compatible with a tensor representation.

    if (img.format().memLayout() != NVCV_MEM_LAYOUT_PL)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format's memory layout must be pitch-linear";
    }

    if (img.format().css() != NVCV_CSS_444)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image format's memory layout must not have sub-sampled planes";
    }

    for (int p = 1; p < fmt.numPlanes(); ++p)
    {
        if (fmt.planePixelType(p) != fmt.planePixelType(0))
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Pixel type of all image planes must be the same";
        }
    }

    NVCVImageData imgData;
    img.exportData(imgData);

    if (imgData.bufferType != NVCV_IMAGE_BUFFER_PITCH_DEVICE)
    {
        throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
            << "Only device-accessible images with pitch-linear data are accepted";
    }

    NVCVImageBufferPitch &imgPitch = imgData.buffer.pitch;

    for (int p = 1; p < imgPitch.numPlanes; ++p)
    {
        if (imgPitch.planes[p].width != imgPitch.planes[0].width
            || imgPitch.planes[p].height != imgPitch.planes[0].height)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "All image planes must have the same dimensions";
        }

        if (imgPitch.planes[p].pitchBytes != imgPitch.planes[0].pitchBytes)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "All image planes must have the same row pitch";
        }

        if (imgPitch.planes[p].buffer <= imgPitch.planes[0].buffer)
        {
            throw Exception(NVCV_ERROR_INVALID_ARGUMENT)
                << "Consecutive image planes must have increasing memory addresses";
        }

        if (p >= 2)
        {
            intptr_t planePitchBytes = reinterpret_cast<const std::byte *>(imgPitch.planes[1].buffer)
                                     - reinterpret_cast<const std::byte *>(imgPitch.planes[0].buffer);

            if (reinterpret_cast<const std::byte *>(imgPitch.planes[p].buffer)
                    - reinterpret_cast<const std::byte *>(imgPitch.planes[p - 1].buffer)
                != planePitchBytes)
            {
                throw Exception(NVCV_ERROR_INVALID_ARGUMENT) << "Image planes must have the same plane pitch";
            }
        }
    }

    // Now fill up tensor data with image data

    tensorData            = {}; // start everything afresh
    tensorData.bufferType = NVCV_TENSOR_BUFFER_PITCH_DEVICE;

    NVCVTensorBufferPitch &tensorPitch = tensorData.buffer.pitch;

    // Infer layout and shape
    std::array<int32_t, 4> bits    = fmt.bpc();
    bool                   sameBPC = true;
    for (int i = 1; i < fmt.numChannels(); ++i)
    {
        if (bits[i] != bits[0])
        {
            sameBPC = false;
            break;
        }
    }

    if (imgPitch.numPlanes == 1)
    {
        if (fmt.numChannels() >= 2 && sameBPC)
        {
            // If same BPC, we can have channels as its own dimension,
            // as all channels have the same type.
            tensorData.layout = NVCV_TENSOR_NHWC;
        }
        else
        {
            tensorData.layout = NVCV_TENSOR_NCHW;
        }
    }
    else
    {
        tensorData.layout = NVCV_TENSOR_NCHW;
    }

    tensorData.ndim = 4;
    if (tensorData.layout == NVCV_TENSOR_NHWC)
    {
        tensorData.shape[0] = 1;
        tensorData.shape[1] = imgPitch.planes[0].height;
        tensorData.shape[2] = imgPitch.planes[0].width;
        tensorData.shape[3] = fmt.numChannels();

        tensorPitch.pitchBytes[3] = fmt.planePixelStrideBytes(0) / fmt.numChannels();
        tensorPitch.pitchBytes[2] = fmt.planePixelStrideBytes(0);
        tensorPitch.pitchBytes[1] = imgPitch.planes[0].pitchBytes;
        tensorPitch.pitchBytes[0] = tensorPitch.pitchBytes[1] * tensorData.shape[1];

        tensorData.dtype = fmt.planePixelType(0).channelType(0).value();
    }
    else
    {
        NVCV_ASSERT(tensorData.layout == NVCV_TENSOR_NCHW);

        tensorData.shape[0] = 1;
        tensorData.shape[1] = imgPitch.numPlanes;
        tensorData.shape[2] = imgPitch.planes[0].height;
        tensorData.shape[3] = imgPitch.planes[0].width;

        tensorPitch.pitchBytes[3] = fmt.planePixelStrideBytes(0);
        tensorPitch.pitchBytes[2] = imgPitch.planes[0].pitchBytes;
        tensorPitch.pitchBytes[1] = tensorPitch.pitchBytes[2] * tensorData.shape[2];
        tensorPitch.pitchBytes[0] = tensorPitch.pitchBytes[1] * tensorData.shape[1];

        tensorData.dtype = fmt.planePixelType(0).value();
    }

    // Finally, assign the pointer to the memory buffer.
    tensorPitch.data = imgPitch.planes[0].buffer;
}

} // namespace nv::cv::priv
