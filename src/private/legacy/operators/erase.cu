/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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

using namespace nv::cv::legacy::helpers;

using namespace nv::cv::legacy::cuda_op;

namespace nvcv = nv::cv;

__device__ int erase_hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

template<typename D, typename Ptr2D, typename Ptr2DI, typename Ptr2DF>
__global__ void erase(Ptr2D img, int imgH, int imgW, Ptr2DI anchorxVec, Ptr2DI anchoryVec, Ptr2DI erasingwVec,
                      Ptr2DI erasinghVec, Ptr2DI erasingcVec, Ptr2DF valuesVec, Ptr2DI imgIdxVec, int channels,
                      int random, unsigned int seed)
{
    unsigned int id        = threadIdx.x + blockIdx.x * blockDim.x;
    int          c         = blockIdx.y;
    int          eraseId   = blockIdx.z;
    int          anchor_x  = *anchorxVec.ptr(0, 0, eraseId);
    int          anchor_y  = *anchoryVec.ptr(0, 0, eraseId);
    int          erasing_w = *erasingwVec.ptr(0, 0, eraseId);
    int          erasing_h = *erasinghVec.ptr(0, 0, eraseId);
    int          erasing_c = *erasingcVec.ptr(0, 0, eraseId);
    float        value     = *valuesVec.ptr(0, 0, eraseId * channels + c);
    int          batchId   = *imgIdxVec.ptr(0, 0, eraseId);
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
void eraseCaller(
    const nvcv::TensorDataAccessPitchImagePlanar &imgs, const nvcv::TensorDataAccessPitchImagePlanar &anchorx,
    const nvcv::TensorDataAccessPitchImagePlanar &anchory, const nvcv::TensorDataAccessPitchImagePlanar &erasingw,
    const nvcv::TensorDataAccessPitchImagePlanar &erasingh, const nvcv::TensorDataAccessPitchImagePlanar &erasingc,
    const nvcv::TensorDataAccessPitchImagePlanar &imgIdx, const nvcv::TensorDataAccessPitchImagePlanar &values,
    int max_eh, int max_ew, int num_erasing_area, bool random, unsigned int seed, cudaStream_t stream)
{
    Ptr2dNHWC<D> src(imgs);

    Ptr2dNHWC<int>   anchorxVec(anchorx);
    Ptr2dNHWC<int>   anchoryVec(anchory);
    Ptr2dNHWC<int>   erasingwVec(erasingw);
    Ptr2dNHWC<int>   erasinghVec(erasingh);
    Ptr2dNHWC<int>   erasingcVec(erasingc);
    Ptr2dNHWC<int>   imgIdxVec(imgIdx);
    Ptr2dNHWC<float> valuesVec(values);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    int  blockSize = (max_eh * max_ew < 1024) ? max_eh * max_ew : 1024;
    int  gridSize  = divUp(max_eh * max_ew, 1024);
    dim3 block(blockSize);
    dim3 grid(gridSize, imgs.numChannels(), num_erasing_area);
    erase<D, Ptr2dNHWC<D>, Ptr2dNHWC<int>, Ptr2dNHWC<float>>
        <<<grid, block, 0, stream>>>(src, imgs.numRows(), imgs.numCols(), anchorxVec, anchoryVec, erasingwVec,
                                     erasinghVec, erasingcVec, valuesVec, imgIdxVec, imgs.numChannels(), random, seed);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nv::cv::legacy::cuda_op {

/*
size_t Erase::calBufferSize(
                DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type, int num_erasing_area)
{
    return num_erasing_area * sizeof(int) * 6 + num_erasing_area * sizeof(float) * 4;
}
*/
// todo support random value && rgb value

ErrorCode Erase::infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                       const ITensorDataPitchDevice &anchor_x, const ITensorDataPitchDevice &anchor_y,
                       const ITensorDataPitchDevice &erasing_w, const ITensorDataPitchDevice &erasing_h,
                       const ITensorDataPitchDevice &erasing_c, const ITensorDataPitchDevice &values,
                       const ITensorDataPitchDevice &imgIdx, int max_eh, int max_ew, bool random, unsigned int seed,
                       bool inplace, cudaStream_t stream)
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

    DataType   anchorx_data_type = GetLegacyDataType(anchor_x.dtype());
    DataFormat anchorx_format    = GetLegacyDataFormat(anchor_x.layout());
    if (anchorx_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid anchor_x DataType " << anchorx_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(anchorx_format == kNHWC || anchorx_format == kHWC))
    {
        LOG_ERROR("Invalid anchor_x DataFormat " << anchorx_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto anchorxAccess = TensorDataAccessPitchImagePlanar::Create(anchor_x);
    if (!anchorxAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    int num_erasing_area = anchorxAccess->numCols();
    if (num_erasing_area < 0)
    {
        LOG_ERROR("Invalid num of erasing area " << num_erasing_area);
        return ErrorCode::INVALID_PARAMETER;
    }

    DataType   anchory_data_type = GetLegacyDataType(anchor_y.dtype());
    DataFormat anchory_format    = GetLegacyDataFormat(anchor_y.layout());
    if (anchory_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid anchor_y DataType " << anchory_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(anchory_format == kNHWC || anchory_format == kHWC))
    {
        LOG_ERROR("Invalid anchor_y DataFormat " << anchory_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto anchoryAccess = TensorDataAccessPitchImagePlanar::Create(anchor_y);
    if (!anchoryAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   erasingw_data_type = GetLegacyDataType(erasing_w.dtype());
    DataFormat erasingw_format    = GetLegacyDataFormat(erasing_w.layout());
    if (erasingw_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid erasing_w DataType " << erasingw_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(erasingw_format == kNHWC || erasingw_format == kHWC))
    {
        LOG_ERROR("Invalid erasing_w DataFormat " << erasingw_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto erasingwAccess = TensorDataAccessPitchImagePlanar::Create(erasing_w);
    if (!erasingwAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   erasingh_data_type = GetLegacyDataType(erasing_h.dtype());
    DataFormat erasingh_format    = GetLegacyDataFormat(erasing_h.layout());
    if (erasingh_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid erasing_h DataType " << erasingh_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(erasingh_format == kNHWC || erasingh_format == kHWC))
    {
        LOG_ERROR("Invalid erasing_h DataFormat " << erasingh_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto erasinghAccess = TensorDataAccessPitchImagePlanar::Create(erasing_h);
    if (!erasinghAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   erasingc_data_type = GetLegacyDataType(erasing_c.dtype());
    DataFormat erasingc_format    = GetLegacyDataFormat(erasing_c.layout());
    if (erasingc_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid erasing_w DataType " << erasingc_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(erasingc_format == kNHWC || erasingc_format == kHWC))
    {
        LOG_ERROR("Invalid erasing_c DataFormat " << erasingc_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto erasingcAccess = TensorDataAccessPitchImagePlanar::Create(erasing_c);
    if (!erasingcAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   imgidx_data_type = GetLegacyDataType(imgIdx.dtype());
    DataFormat imgidx_format    = GetLegacyDataFormat(imgIdx.layout());
    if (imgidx_data_type != kCV_32S)
    {
        LOG_ERROR("Invalid imgIdx DataType " << imgidx_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(imgidx_format == kNHWC || imgidx_format == kHWC))
    {
        LOG_ERROR("Invalid imgIdx DataFormat " << imgidx_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto imgIdxAccess = TensorDataAccessPitchImagePlanar::Create(imgIdx);
    if (!imgIdxAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
    }

    DataType   values_data_type = GetLegacyDataType(values.dtype());
    DataFormat values_format    = GetLegacyDataFormat(values.layout());
    if (values_data_type != kCV_32F)
    {
        LOG_ERROR("Invalid values DataType " << values_data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (!(values_format == kNHWC || values_format == kHWC))
    {
        LOG_ERROR("Invalid values DataFormat " << values_format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }
    auto valuesAccess = TensorDataAccessPitchImagePlanar::Create(values);
    if (!valuesAccess)
    {
        return ErrorCode::INVALID_DATA_TYPE;
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

    /*
    void *inputImgs;
    if(inplace)
    {
        inputImgs = inputs[0];
    }
    else
    {
        inputImgs = outputs[0];
    }

    int max_eh = 0, max_ew = 0;
    for(int i = 0; i < num_erasing_area; i++)
    {
        int eh = erasing_h[i], ew = erasing_w[i];
        if(eh * ew > max_eh * max_ew)
        {
            max_eh = eh;
            max_ew = ew;
        }
    }
    */

    typedef void (*erase_t)(
        const TensorDataAccessPitchImagePlanar &imgs, const TensorDataAccessPitchImagePlanar &anchorx,
        const TensorDataAccessPitchImagePlanar &anchory, const TensorDataAccessPitchImagePlanar &erasingw,
        const TensorDataAccessPitchImagePlanar &erasingh, const TensorDataAccessPitchImagePlanar &erasingc,
        const TensorDataAccessPitchImagePlanar &imgIdx, const TensorDataAccessPitchImagePlanar &values, int max_eh,
        int max_ew, int num_erasing_area, bool random, unsigned int seed, cudaStream_t stream);

    static const erase_t funcs[6] = {eraseCaller<uchar>, eraseCaller<char>, eraseCaller<ushort>,
                                     eraseCaller<short>, eraseCaller<int>,  eraseCaller<float>};

    if (inplace)
        funcs[data_type](*inAccess, *anchorxAccess, *anchoryAccess, *erasingwAccess, *erasinghAccess, *erasingcAccess,
                         *imgIdxAccess, *valuesAccess, max_eh, max_ew, num_erasing_area, random, seed, stream);
    else
        funcs[data_type](*outAccess, *anchorxAccess, *anchoryAccess, *erasingwAccess, *erasinghAccess, *erasingcAccess,
                         *imgIdxAccess, *valuesAccess, max_eh, max_ew, num_erasing_area, random, seed, stream);

    return SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
