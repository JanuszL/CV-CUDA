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

#include <common/NvDecoder.h>
#include <common/TestUtils.h>
#include <cuda_runtime_api.h>
#include <getopt.h>
#include <math.h>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <operators/OpCustomCrop.hpp>
#include <operators/OpResize.hpp>

/**
 * @brief Crop and Resize sample app.
 *
 * The Crop and Resize is a simple pipeline which demonstrates usage of
 * CVCuda Tensor along with a few operators.
 *
 * Input Batch Tensor -> Crop -> Resize -> WriteImage
 */

/**
 * @brief Utility to show usage of sample app
 *
 **/
void showUsage()
{
    std::cout << "usage: ./nvcv_cropandresize_app -i <image file path or  image directory -b <batch size>" << std::endl;
}

/**
 * @brief Utility to parse the command line arguments
 *
 **/
int ParseArgs(int argc, char *argv[], std::string &imagePath, uint32_t &batchSize)
{
    static struct option long_options[] = {
        {     "help",       no_argument, 0, 'h'},
        {"imagePath", required_argument, 0, 'i'},
        {    "batch", required_argument, 0, 'b'},
        {          0,                 0, 0,   0}
    };

    int long_index = 0;
    int opt        = 0;
    while ((opt = getopt_long(argc, argv, "hi:b:", long_options, &long_index)) != -1)
    {
        switch (opt)
        {
        case 'h':
            showUsage();
            return -1;
            break;
        case 'i':
            imagePath = optarg;
            break;
        case 'b':
            batchSize = std::stoi(optarg);
            break;
        case ':':
            showUsage();
            return -1;
        default:
            break;
        }
    }
    std::ifstream imageFile(imagePath);
    if (!imageFile.good())
    {
        showUsage();
        std::cerr << "Image path '" + imagePath + "' does not exist\n";
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    // Default parameters
    std::string imagePath = "./samples/assets/tabby_tiger_cat.jpg";
    uint32_t    batchSize = 1;

    // Parse the command line paramaters to override the default parameters
    int retval = ParseArgs(argc, argv, imagePath, batchSize);
    if (retval != 0)
    {
        return retval;
    }

    // NvJpeg is used to decode the images to the color format required.
    // Since we need a contiguous buffer for batched input, a buffer is
    // preallocated based on the  maximum image dimensions and  batch size
    // for NvJpeg to write into.

    // Note : The maximum input image dimensions needs to be updated in case
    // of testing with different test images

    int maxImageWidth  = 720;
    int maxImageHeight = 720;
    int maxChannels    = 3;

    // Create the cuda stream
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocating memory for RGBI input image batch of uint8_t data type
    // without padding since NvDecode utility currently doesnt support
    // Padded buffers.

    nv::cv::TensorDataPitchDevice::Buffer inBuf;
    inBuf.pitchBytes[3] = sizeof(uint8_t);
    inBuf.pitchBytes[2] = maxChannels * inBuf.pitchBytes[3];
    inBuf.pitchBytes[1] = maxImageWidth * inBuf.pitchBytes[2];
    inBuf.pitchBytes[0] = maxImageHeight * inBuf.pitchBytes[1];
    CHECK_CUDA_ERROR(cudaMallocAsync(&inBuf.data, batchSize * inBuf.pitchBytes[0], stream));

    // Calculate the requirements for the RGBI uint8_t Tensor which include
    // pitch bytes, alignment, shape  and tensor layout
    nv::cv::Tensor::Requirements inReqs
        = nv::cv::Tensor::CalcRequirements(batchSize, {maxImageWidth, maxImageHeight}, nv::cv::FMT_RGB8);

    // Create a tensor buffer to store the data pointer and pitch bytes for each plane
    nv::cv::TensorDataPitchDevice inData(nv::cv::TensorShape{inReqs.shape, inReqs.ndim, inReqs.layout},
                                         nv::cv::PixelType{inReqs.dtype}, inBuf);

    // TensorWrapData allows for interoperation of external tensor representations with CVCUDA Tensor.
    nv::cv::TensorWrapData inTensor(inData);

    // NvJpeg is used to load the images to create a batched input device buffer.
    uint8_t             *gpuInput = static_cast<uint8_t *>(inBuf.data);
    // The total images is set to the same value as batch size for testing
    uint32_t             totalImages = batchSize;
    // Format in which the decoded output will be saved
    nvjpegOutputFormat_t outputFormat = NVJPEG_OUTPUT_RGBI;

    NvDecode(imagePath, batchSize, totalImages, outputFormat, gpuInput);

    // The input buffer is now ready to be used by the operators

    // Set parameters for Crop and Resize
    // ROI dimensions to crop in the input image
    int cropX      = 150;
    int cropY      = 50;
    int cropWidth  = 400;
    int cropHeight = 300;

    // Set the resize dimensions
    int resizeWidth  = 320;
    int resizeHeight = 240;

    //  Initialize the CVCUDA ROI struct
    NVCVRectI crpRect = {cropX, cropY, cropWidth, cropHeight};

    // Create a CVCUDA Tensor based on the crop window size.
    nv::cv::Tensor cropTensor(batchSize, {cropWidth, cropHeight}, nv::cv::FMT_RGB8);
    // Create a CVCUDA Tensor based on resize dimensions
    nv::cv::Tensor resizedTensor(batchSize, {resizeWidth, resizeHeight}, nv::cv::FMT_RGB8);

#ifdef PROFILE_SAMPLE
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    // Initialize crop operator
    nv::cvop::CustomCrop cropOp;
    // Initialize resize operator
    nv::cvop::Resize     resizeOp;

    // Executes the CustomCrop operation on the given cuda stream
    cropOp(stream, inTensor, cropTensor, crpRect);

    // Resize operator can now be enqueued into the same stream
    resizeOp(stream, cropTensor, resizedTensor, NVCV_INTERP_LINEAR);

#ifdef PROFILE_SAMPLE
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float operatorms = 0;
    cudaEventElapsedTime(&operatorms, start, stop);
    std::cout << "Time for Crop and Resize : " << operatorms << " ms" << std::endl;
#endif

    // Copy the buffer to CPU and write resized image into .bmp file
    WriteRGBITensor(resizedTensor, stream);

    // Clean up
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}