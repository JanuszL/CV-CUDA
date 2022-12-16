/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include "../CvCudaLegacy.h"
#include "../CvCudaLegacyHelpers.hpp"

#include "../CvCudaUtils.cuh"
#include "cub/cub.cuh"

using namespace nv::cv::legacy::helpers;

using namespace nv::cv::legacy::cuda_op;

namespace nvcv = nv::cv;

static __device__ int erase_hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

template<typename D>
__global__ void erase(nvcv::cuda::Tensor4DWrap<D> img, int imgH, int imgW, nvcv::cuda::Tensor1DWrap<int2> anchorVec,
                      nvcv::cuda::Tensor1DWrap<int3> erasingVec, nvcv::cuda::Tensor1DWrap<float> valuesVec,
                      nvcv::cuda::Tensor1DWrap<int> imgIdxVec, int channels, int random, unsigned int seed)
{
    unsigned int id        = threadIdx.x + blockIdx.x * blockDim.x;
    int          c         = blockIdx.y;
    int          eraseId   = blockIdx.z;
    int          anchor_x  = (*anchorVec.ptr(eraseId)).x;
    int          anchor_y  = (*anchorVec.ptr(eraseId)).y;
    int          erasing_w = (*erasingVec.ptr(eraseId)).x;
    int          erasing_h = (*erasingVec.ptr(eraseId)).y;
    int          erasing_c = (*erasingVec.ptr(eraseId)).z;
    float        value     = *valuesVec.ptr(eraseId * channels + c);
    int          batchId   = *imgIdxVec.ptr(eraseId);
    if (id < erasing_h * erasing_w && (0x1 & (erasing_c >> c)) == 1)
    {
        int x = id % erasing_w;
        int y = id / erasing_w;
        if (anchor_x + x < imgW && anchor_y + y < imgH)
        {
            if (random)
            {
                unsigned int hashValue = seed + threadIdx.x
                                       + 0x26AD0C9 * blockDim.x * blockDim.y * blockDim.z * (blockIdx.x + 1)
                                             * (blockIdx.y + 1) * (blockIdx.z + 1);
                *img.ptr(batchId, anchor_y + y, anchor_x + x, c)
                    = nv::cv::cuda::SaturateCast<D>(erase_hash(hashValue) % 256);
            }
            else
            {
                *img.ptr(batchId, anchor_y + y, anchor_x + x, c) = nv::cv::cuda::SaturateCast<D>(value);
            }
        }
    }
}

template<typename D>
void eraseCaller(const nvcv::ITensorDataPitchDevice &imgs, const nvcv::ITensorDataPitchDevice &anchor,
                 const nvcv::ITensorDataPitchDevice &erasing, const nvcv::ITensorDataPitchDevice &imgIdx,
                 const nvcv::ITensorDataPitchDevice &values, int max_eh, int max_ew, int num_erasing_area, bool random,
                 unsigned int seed, int rows, int cols, int channels, cudaStream_t stream)
{
    nvcv::cuda::Tensor4DWrap<D> src(imgs);

    nvcv::cuda::Tensor1DWrap<int2>  anchorVec(anchor);
    nvcv::cuda::Tensor1DWrap<int3>  erasingVec(erasing);
    nvcv::cuda::Tensor1DWrap<int>   imgIdxVec(imgIdx);
    nvcv::cuda::Tensor1DWrap<float> valuesVec(values);

    int  blockSize = (max_eh * max_ew < 1024) ? max_eh * max_ew : 1024;
    int  gridSize  = divUp(max_eh * max_ew, 1024);
    dim3 block(blockSize);
    dim3 grid(gridSize, channels, num_erasing_area);
    erase<D><<<grid, block, 0, stream>>>(src, rows, cols, anchorVec, erasingVec, valuesVec, imgIdxVec, channels, random,
                                         seed);
}

struct MaxWH
{
    __device__ __forceinline__ int3 operator()(const int3 &a, const int3 &b) const
    {
        return int3{max(a.x, b.x), max(a.y, b.y), 0};
    }
};

namespace nv::cv::legacy::cuda_op {

Erase::Erase(DataShape max_input_shape, DataShape max_output_shape, int num_erasing_area)
    : CudaBaseOp(max_input_shape, max_output_shape)
    , d_max_values(nullptr)
    , temp_storage(nullptr)
{
    cudaError_t err = cudaMalloc(&d_max_values, sizeof(int3));
    if (err != cudaSuccess)
    {
        LOG_ERROR("CUDA memory allocation error of size: " << sizeof(int3));
        throw std::runtime_error("CUDA memory allocation error!");
    }

    max_num_erasing_area = num_erasing_area;
    if (max_num_erasing_area < 0)
    {
        cudaFree(d_max_values);
        LOG_ERROR("Invalid num of erasing area" << max_num_erasing_area);
        throw std::runtime_error("Parameter error!");
    }
    temp_storage  = NULL;
    storage_bytes = 0;
    MaxWH mwh;
    int3  init = {0, 0, 0};
    cub::DeviceReduce::Reduce(temp_storage, storage_bytes, (int3 *)nullptr, (int3 *)nullptr, max_num_erasing_area, mwh,
                              init);

    err = cudaMalloc(&temp_storage, storage_bytes);
    if (err != cudaSuccess)
    {
        cudaFree(d_max_values);
        LOG_ERROR("CUDA memory allocation error of size: " << storage_bytes);
        throw std::runtime_error("CUDA memory allocation error!");
    }
}

Erase::~Erase()
{
    cudaError_t err0 = cudaFree(d_max_values);
    cudaError_t err1 = cudaFree(temp_storage);
    if (err0 != cudaSuccess || err1 != cudaSuccess)
    {
        LOG_ERROR("CUDA memory free error, possible memory leak!");
    }
    d_max_values = nullptr;
    temp_storage = nullptr;
}

ErrorCode Erase::infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                       const ITensorDataPitchDevice &anchor, const ITensorDataPitchDevice &erasing,
                       const ITensorDataPitchDevice &values, const ITensorDataPitchDevice &imgIdx, bool random,
                       unsigned int seed, bool inplace, cudaStream_t stream)
{
    DataFormat format    = GetLegacyDataFormat(inData.layout());
    DataType   data_type = GetLegacyDataType(inData.dtype());

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType anchor_data_type = GetLegacyDataType(anchor.dtype());
    if (anchor_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid anchor DataType " << anchor_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int anchor_dim = anchor.layout().ndim();
    if (anchor_dim != 1)
    {
        LOG_ERROR("Invalid anchor Dim " << anchor_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int num_erasing_area = anchor.shape()[0];
    if (num_erasing_area < 0)
    {
        LOG_ERROR("Invalid num of erasing area " << num_erasing_area);
        return ErrorCode::INVALID_PARAMETER;
    }
    if (num_erasing_area > max_num_erasing_area)
    {
        LOG_ERROR("Invalid num of erasing area " << num_erasing_area);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType erasing_data_type = GetLegacyDataType(erasing.dtype());
    if (erasing_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid erasing_w DataType " << erasing_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int erasing_dim = erasing.layout().ndim();
    if (erasing_dim != 1)
    {
        LOG_ERROR("Invalid erasing Dim " << erasing_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType imgidx_data_type = GetLegacyDataType(imgIdx.dtype());
    if (imgidx_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid imgIdx DataType " << imgidx_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int imgidx_dim = imgIdx.layout().ndim();
    if (imgidx_dim != 1)
    {
        LOG_ERROR("Invalid imgIdx Dim " << imgidx_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType values_data_type = GetLegacyDataType(values.dtype());
    if (values_data_type != kCV_32F)
    {
        LOG_ERROR("Invalid values DataType " << values_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    int values_dim = values.layout().ndim();
    if (values_dim != 1)
    {
        LOG_ERROR("Invalid values Dim " << values_dim);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = TensorDataAccessPitchImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessPitchImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    if (!inplace)
    {
        for (uint32_t i = 0; i < inAccess->numSamples(); ++i)
        {
            void *inSampData  = inAccess->sampleData(i);
            void *outSampData = outAccess->sampleData(i);

            checkCudaErrors(cudaMemcpy2DAsync(outSampData, outAccess->rowPitchBytes(), inSampData,
                                              inAccess->rowPitchBytes(),
                                              inAccess->numCols() * inAccess->colPitchBytes(), inAccess->numRows(),
                                              cudaMemcpyDeviceToDevice, stream));
        }
    }

    if (num_erasing_area == 0)
    {
        return SUCCESS;
    }

    int3 *d_erasing = (int3 *)erasing.data();
    int3  h_max_values;
    MaxWH maxwh;
    int3  init = {0, 0, 0};

    cub::DeviceReduce::Reduce(temp_storage, storage_bytes, d_erasing, d_max_values, num_erasing_area, maxwh, init,
                              stream);
    checkCudaErrors(cudaMemcpyAsync(&h_max_values, d_max_values, sizeof(int3), cudaMemcpyDeviceToHost, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));

    int max_ew = h_max_values.x, max_eh = h_max_values.y;

    // All areas as empty? Weird, but valid nonetheless.
    if (max_ew == 0 || max_eh == 0)
    {
        return SUCCESS;
    }

    typedef void (*erase_t)(const ITensorDataPitchDevice &imgs, const ITensorDataPitchDevice &anchor,
                            const ITensorDataPitchDevice &erasing, const ITensorDataPitchDevice &imgIdx,
                            const ITensorDataPitchDevice &values, int max_eh, int max_ew, int num_erasing_area,
                            bool random, unsigned int seed, int rows, int cols, int channels, cudaStream_t stream);

    static const erase_t funcs[6] = {eraseCaller<uchar>, eraseCaller<char>, eraseCaller<ushort>,
                                     eraseCaller<short>, eraseCaller<int>,  eraseCaller<float>};

    if (inplace)
        funcs[data_type](inData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed,
                         inAccess->numRows(), inAccess->numCols(), inAccess->numChannels(), stream);
    else
        funcs[data_type](outData, anchor, erasing, imgIdx, values, max_eh, max_ew, num_erasing_area, random, seed,
                         outAccess->numRows(), outAccess->numCols(), outAccess->numChannels(), stream);

    return SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
