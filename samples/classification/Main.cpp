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

#include "ClassificationUtils.hpp"

#include <common/NvDecoder.h>
#include <common/TRTUtils.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>
#include <operators/OpConvertTo.hpp>
#include <operators/OpNormalize.hpp>
#include <operators/OpReformat.hpp>
#include <operators/OpResize.hpp>

#include <fstream>
#include <iostream>
#include <numeric>

/**
 * @brief Image classification sample.
 *
 * The image classification sample uses Resnet50 based model trained on Imagenet
 * The sample app includes preprocessing, inference and post process stages
 *
 * In the preprocessing The following operations are performed
 * Resize -> DataType COnvert -> Normalize -> Interleaved to Planar
 * The inference pipeline is run using TensorRT
 * The post processing involves running a softmax to normalize the scores to 0-1
 * and sorting the scores to find the TopN classificaion results.
 *
 */

void PreProcess(nv::cv::TensorWrapData &inTensor, int maxImageWidth, int maxImageHeight, uint32_t batchSize,
                int inputLayerWidth, int inputLayerHeight, cudaStream_t m_cudaStream, nv::cv::TensorWrapData &outTensor)
{
    // Resize
    nv::cv::Tensor   resizedTensor(batchSize, {inputLayerWidth, inputLayerHeight}, nv::cv::FMT_RGB8);
    nv::cvop::Resize resizeOp;
    resizeOp(m_cudaStream, inTensor, resizedTensor, NVCV_INTERP_LINEAR);

    // Convert U8 - F32. Apply scale 1/255f.
    nv::cv::Tensor      tempTensor(batchSize, {inputLayerWidth, inputLayerHeight}, nv::cv::FMT_RGBf32);
    nv::cvop::ConvertTo convertOp;
    convertOp(m_cudaStream, resizedTensor, tempTensor, 1.0f / 255.f, 0.0f);

    // Apply std deviation
    nv::cv::Tensor::Requirements reqsScale       = nv::cv::Tensor::CalcRequirements(1, {1, 1}, nv::cv::FMT_RGBf32);
    int64_t                      scaleBufferSize = CalcTotalSizeBytes(nv::cv::Requirements{reqsScale.mem}.deviceMem());
    nv::cv::TensorDataPitchDevice::Buffer bufScale;
    std::copy(reqsScale.pitchBytes, reqsScale.pitchBytes + NVCV_TENSOR_MAX_NDIM, bufScale.pitchBytes);
    CHECK_CUDA_ERROR(cudaMalloc(&bufScale.data, scaleBufferSize));
    nv::cv::TensorDataPitchDevice scaleIn(nv::cv::TensorShape{reqsScale.shape, reqsScale.ndim, reqsScale.layout},
                                          nv::cv::PixelType{reqsScale.dtype}, bufScale);
    nv::cv::TensorWrapData        scaleTensor(scaleIn);

    // Apply mean shift
    nv::cv::TensorDataPitchDevice::Buffer bufBase;
    nv::cv::Tensor::Requirements          reqsBase = nv::cv::Tensor::CalcRequirements(1, {1, 1}, nv::cv::FMT_RGBf32);
    int64_t baseBufferSize                         = CalcTotalSizeBytes(nv::cv::Requirements{reqsBase.mem}.deviceMem());
    std::copy(reqsBase.pitchBytes, reqsBase.pitchBytes + NVCV_TENSOR_MAX_NDIM, bufBase.pitchBytes);
    CHECK_CUDA_ERROR(cudaMalloc(&bufBase.data, baseBufferSize));
    nv::cv::TensorDataPitchDevice baseIn(nv::cv::TensorShape{reqsBase.shape, reqsBase.ndim, reqsBase.layout},
                                         nv::cv::PixelType{reqsBase.dtype}, bufBase);
    nv::cv::TensorWrapData        baseTensor(baseIn);

    const auto *baseData  = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(scaleTensor.exportData());
    const auto *scaleData = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(baseTensor.exportData());

    // Preprocessing parameters
    float    scale[3] = {0.229, 0.224, 0.225};
    float    base[3]  = {0.485f, 0.456f, 0.406f};
    uint32_t flags    = NVCV_OP_NORMALIZE_SCALE_IS_STDDEV;
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(scaleData->data(), scale, 3 * sizeof(float), cudaMemcpyHostToDevice, m_cudaStream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(baseData->data(), base, 3 * sizeof(float), cudaMemcpyHostToDevice, m_cudaStream));

    nv::cv::Tensor normTensor(batchSize, {inputLayerWidth, inputLayerHeight}, nv::cv::FMT_RGBf32);

    // Normalize
    nv::cvop::Normalize normOp;
    normOp(m_cudaStream, tempTensor, baseTensor, scaleTensor, normTensor, 1.0f, 0.0f, 0.0f, flags);

    // Interleaved to planar
    nv::cvop::Reformat reformatOp;
    reformatOp(m_cudaStream, normTensor, outTensor);
}

void PostProcess(float *outputDeviceBuffer, std::vector<std::vector<float>> &sortedScores,
                 std::vector<std::vector<int>> &sortedIndices, cudaStream_t stream)
{
    uint32_t batchSize  = sortedScores.size();
    uint32_t numClasses = sortedScores[0].size();

    for (int i = 0; i < batchSize; i++)
    {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(sortedScores[i].data(), outputDeviceBuffer + i * numClasses,
                                         numClasses * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (int i = 0; i < batchSize; i++)
    {
        std::transform(sortedScores[i].begin(), sortedScores[i].end(), sortedScores[i].begin(),
                       [](float val) { return std::exp(val); });

        auto sum = std::accumulate(sortedScores[i].begin(), sortedScores[i].end(), 0.0);
        std::transform(sortedScores[i].begin(), sortedScores[i].end(), sortedScores[i].begin(),
                       [sum](float val) { return val / sum; });
        std::iota(sortedIndices[i].begin(), sortedIndices[i].end(), 0);
        std::sort(sortedIndices[i].begin(), sortedIndices[i].end(),
                  [&sortedScores, i](int i1, int i2) { return sortedScores[i][i1] > sortedScores[i][i2]; });
    }
}

int main(int argc, char *argv[])
{
    // Default parameters
    std::string modelPath = "./engines/resnet50.engine";
    std::string imagePath = "./samples/assets/tabby_tiger_cat.jpg";
    std::string labelPath = "./engines/imagenet-classes.txt";
    uint32_t    batchSize = 1;

    static struct option long_options[] = {
        {     "help",       no_argument, 0, 'h'},
        {   "engine", required_argument, 0, 'e'},
        {"labelPath", required_argument, 0, 'l'},
        {"imagePath", required_argument, 0, 'i'},
        {    "batch", required_argument, 0, 'b'},
        {          0,                 0, 0,   0}
    };

    int long_index = 0;
    int opt        = 0;
    while ((opt = getopt_long(argc, argv, "he:l:i:b:", long_options, &long_index)) != -1)
    {
        switch (opt)
        {
        case 'h':
            showUsage();
            return -1;
            break;
        case 'e':
            modelPath = optarg;
            break;
        case 'l':
            labelPath = optarg;
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

    std::ifstream modelFile(modelPath);
    if (!modelFile.good())
    {
        showUsage();
        throw std::runtime_error("Model path '" + modelPath + "' does not exist\n");
    }
    std::ifstream imageFile(imagePath);
    if (!imageFile.good())
    {
        showUsage();
        throw std::runtime_error("Image path '" + imagePath + "' does not exist\n");
    }
    std::ifstream labelFile(labelPath);
    if (!labelFile.good())
    {
        showUsage();
        throw std::runtime_error("Label path '" + labelPath + "' does not exist\n");
    }

    // Setup NvJpeg to load the image file or image directory
    std::vector<nvjpegImage_t> iout;
    std::vector<int>           widths, heights;
    uint32_t                   totalImages  = batchSize;
    nvjpegOutputFormat_t       outputFormat = NVJPEG_OUTPUT_BGRI;
    widths.resize(batchSize);
    heights.resize(batchSize);
    iout.resize(batchSize);

    // Maximum dimension of images
    int maxImageWidth  = 720;
    int maxImageHeight = 720;
    int maxChannels    = 3;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Allocating memory for input image batch
    nv::cv::TensorDataPitchDevice::Buffer inBuf;
    inBuf.pitchBytes[3] = sizeof(uint8_t);
    inBuf.pitchBytes[2] = maxChannels * inBuf.pitchBytes[3];
    inBuf.pitchBytes[1] = maxImageWidth * inBuf.pitchBytes[2];
    inBuf.pitchBytes[0] = maxImageHeight * inBuf.pitchBytes[1];
    CHECK_CUDA_ERROR(cudaMallocAsync(&inBuf.data, batchSize * inBuf.pitchBytes[0], stream));

    nv::cv::Tensor::Requirements inReqs
        = nv::cv::Tensor::CalcRequirements(batchSize, {maxImageWidth, maxImageHeight}, nv::cv::FMT_RGB8);

    nv::cv::TensorDataPitchDevice inData(nv::cv::TensorShape{inReqs.shape, inReqs.ndim, inReqs.layout},
                                         nv::cv::PixelType{inReqs.dtype}, inBuf);

    nv::cv::TensorWrapData inTensor(inData);

    uint8_t *gpuInput = static_cast<uint8_t *>(inBuf.data);

    NvDecode(imagePath, batchSize, totalImages, outputFormat, iout, gpuInput, widths, heights);

    // Setup TRT Backend
    std::unique_ptr<TRTBackend> trtBackend;
    trtBackend.reset(new TRTBackend(modelPath.c_str()));

    auto numBindings = trtBackend->getBlobCount();
    if (numBindings != 2)
    {
        printf("Number of bindings should be 2\n");
        return -1;
    }

    // Initialize TensorRT Buffers
    std::vector<void *> buffers;
    buffers.resize(numBindings);

    // Get network dimensions
    TRTBackendBlobSize inputDims, outputDims;
    uint32_t           inputBindingIndex, outputBindingIndex;
    for (int i = 0; i < numBindings; i++)
    {
        if (trtBackend->bindingIsInput(i))
        {
            inputDims         = trtBackend->getTRTBackendBlobSize(i);
            inputBindingIndex = i;
        }
        else
        {
            outputDims         = trtBackend->getTRTBackendBlobSize(i);
            outputBindingIndex = i;
        }
    }

    // Allocate input layer buffer
    nv::cv::Tensor::Requirements reqsInputLayer
        = nv::cv::Tensor::CalcRequirements(batchSize, {inputDims.width, inputDims.height}, nv::cv::FMT_RGBf32p);
    int64_t inputLayerSize = CalcTotalSizeBytes(nv::cv::Requirements{reqsInputLayer.mem}.deviceMem());
    nv::cv::TensorDataPitchDevice::Buffer bufInputLayer;
    std::copy(reqsInputLayer.pitchBytes, reqsInputLayer.pitchBytes + NVCV_TENSOR_MAX_NDIM, bufInputLayer.pitchBytes);
    CHECK_CUDA_ERROR(cudaMalloc(&bufInputLayer.data, inputLayerSize));
    nv::cv::TensorDataPitchDevice inputLayerTensorData(
        nv::cv::TensorShape{reqsInputLayer.shape, reqsInputLayer.ndim, reqsInputLayer.layout},
        nv::cv::PixelType{reqsInputLayer.dtype}, bufInputLayer);
    nv::cv::TensorWrapData inputLayerTensor(inputLayerTensorData);

    // Allocate output layer buffer
    nv::cv::Tensor::Requirements reqsOutputLayer
        = nv::cv::Tensor::CalcRequirements(batchSize, {outputDims.width, 1}, nv::cv::FMT_RGBf32p);
    int64_t outputLayerSize = CalcTotalSizeBytes(nv::cv::Requirements{reqsOutputLayer.mem}.deviceMem());
    nv::cv::TensorDataPitchDevice::Buffer bufOutputLayer;
    std::copy(reqsOutputLayer.pitchBytes, reqsOutputLayer.pitchBytes + NVCV_TENSOR_MAX_NDIM, bufOutputLayer.pitchBytes);
    CHECK_CUDA_ERROR(cudaMalloc(&bufOutputLayer.data, outputLayerSize));
    nv::cv::TensorDataPitchDevice outputLayerTensorData(
        nv::cv::TensorShape{reqsOutputLayer.shape, reqsOutputLayer.ndim, reqsOutputLayer.layout},
        nv::cv::PixelType{reqsOutputLayer.dtype}, bufOutputLayer);
    nv::cv::TensorWrapData outputLayerTensor(outputLayerTensorData);

    // Preprocess input
    PreProcess(inTensor, maxImageWidth, maxImageHeight, batchSize, inputDims.width, inputDims.height, stream,
               inputLayerTensor);

    // Run Inference
    const auto *inputData  = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(inputLayerTensor.exportData());
    const auto *outputData = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(outputLayerTensor.exportData());

    buffers[inputBindingIndex]  = inputData->data();
    buffers[outputBindingIndex] = outputData->data();

    trtBackend->infer(&buffers[inputBindingIndex], batchSize, stream);

    // Post Process
    uint32_t                        numClasses = outputDims.width;
    std::vector<std::vector<float>> sortedScores(batchSize, std::vector<float>(numClasses));
    std::vector<std::vector<int>>   sortedIndices(batchSize, std::vector<int>(numClasses));
    PostProcess((float *)outputData->data(), sortedScores, sortedIndices, stream);

    // Display Results
    DisplayResults(sortedScores, sortedIndices, labelPath);

    // Clean up
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}
