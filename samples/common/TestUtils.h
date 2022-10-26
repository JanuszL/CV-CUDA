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

#ifndef NVCV_TESTUTILS_H
#define NVCV_TESTUTILS_H

#include "NvDecoder.h"

#include <cuda_runtime_api.h>
#include <nvcv/Tensor.hpp>

#define PROFILE_SAMPLE

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

void WriteRGBITensor(nv::cv::Tensor &inTensor, cudaStream_t &stream)
{
    const auto *srcData = dynamic_cast<const nv::cv::ITensorDataPitchDevice *>(inTensor.exportData());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    int bufferSize = srcData->pitchBytes(0);
    int pitchBytes = srcData->pitchBytes(1);
    int height     = inTensor.shape()[1];
    int width      = inTensor.shape()[2];
    int batchSize  = inTensor.shape()[0];

    for (int b = 0; b < batchSize; b++)
    {
        std::ostringstream ossIn;
        ossIn << "./cvcudatest_" << b << ".bmp";
        writeBMPi(ossIn.str().c_str(), (const unsigned char *)srcData->data() + bufferSize * b, pitchBytes, width,
                  height);
    }
}

#endif // NVCV_TESTUTILS_H
