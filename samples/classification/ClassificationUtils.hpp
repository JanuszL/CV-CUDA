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

#include <cuda_runtime_api.h>
#include <getopt.h>
#include <math.h>
#include <nvcv/Image.hpp>
#include <nvcv/Tensor.hpp>
#include <nvcv/TensorDataAccess.hpp>

#include <fstream>
#include <iostream>
#include <numeric>

/**
 * @brief Utiltities for Image classification
 *
 */

#define TOPN 5

inline void CheckCudaError(cudaError_t code, const char *file, const int line)
{
    if (code != cudaSuccess)
    {
        const char       *errorMessage = cudaGetErrorString(code);
        const std::string message      = "CUDA error returned at " + std::string(file) + ":" + std::to_string(line)
                                  + ", Error code: " + std::to_string(code) + " (" + std::string(errorMessage) + ")";
        throw std::runtime_error(message);
    }
}

#define CHECK_CUDA_ERROR(val)                      \
    {                                              \
        CheckCudaError((val), __FILE__, __LINE__); \
    }

/**
 * @brief Utility to get the class names from the labels file
 *
 **/
std::vector<std::string> getClassLabels(const std::string &labelsFilePath)
{
    std::vector<std::string> classes;
    std::ifstream            labelsFile(labelsFilePath);
    if (!labelsFile.good())
    {
        throw std::runtime_error("ERROR: Invalid Labels File Path\n");
    }
    std::string classLabel;
    while (std::getline(labelsFile, classLabel))
    {
        classes.push_back(classLabel);
    }
    return classes;
}

/**
 * @brief Display the TopN classification results
 *
 **/
void DisplayResults(std::vector<std::vector<float>> &scores, std::vector<std::vector<int>> &indices,
                    std::string labelPath)
{
    auto classes = getClassLabels(labelPath);
    for (int i = 0; i < scores.size(); i++)
    {
        printf("\nClassification results for batch %d \n", i);
        for (int j = 0; j < TOPN; j++)
        {
            auto index = indices[i][j];
            printf("Class : %s , Score : %f\n", classes[index].c_str(), scores[i][index]);
        }
    }
}

/**
 * @brief Utility docs
 *
 **/
void showUsage()
{
    std::cout << "usage: ./nvcv_classification_app -e <tensorrt engine path> -i <image file path or  image directory "
                 "path> -l <labels file path> -b <batch size>"
              << std::endl;
}

/**
 * @brief Utility to parse the command line arguments
 *
 **/
int ParseArgs(int argc, char *argv[], std::string &modelPath, std::string &imagePath, std::string &labelPath,
              uint32_t &batchSize)
{
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
        std::cerr << "Model path '" + modelPath + "' does not exist\n";
        return -1;
    }
    std::ifstream imageFile(imagePath);
    if (!imageFile.good())
    {
        showUsage();
        std::cerr << "Image path '" + modelPath + "' does not exist\n";
        return -1;
    }
    std::ifstream labelFile(labelPath);
    if (!labelFile.good())
    {
        showUsage();
        std::cerr << "Label path '" + modelPath + "' does not exist\n";
        return -1;
    }
    return 0;
}
