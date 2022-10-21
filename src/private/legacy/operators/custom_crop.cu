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

#include <nvcv/IImage.hpp>
#include <nvcv/IImageData.hpp>
#include <nvcv/ITensorData.hpp>

#include <cstdio>

using namespace nv::cv::legacy::cuda_op;
using namespace nv::cv::legacy::helpers;

namespace nvcv = nv::cv;

template<typename Ptr2D>
__global__ void custom_crop_kernel(const Ptr2D src, Ptr2D dst, int start_x, int start_y)
{
    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    if (x >= dst.cols || y >= dst.rows)
        return;

    *dst.ptr(batch_idx, y, x) = *src.ptr(batch_idx, y + start_y, x + start_x);
}

template<typename T>
void customCrop(const nvcv::TensorDataAccessPitchImagePlanar &inData,
                const nvcv::TensorDataAccessPitchImagePlanar &outData, NVCVRectI roi, cudaStream_t stream)
{
    int          cols       = roi.width;
    int          rows       = roi.height;
    const int    batch_size = outData.numSamples();
    Ptr2dNHWC<T> src_ptr(inData);
    Ptr2dNHWC<T> dst_ptr(outData, cols, rows);

    dim3 block(16, 16);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), batch_size);

    custom_crop_kernel<Ptr2dNHWC<T>><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, roi.x, roi.y);
    checkKernelErrors();
}

namespace nv::cv::legacy::cuda_op {

size_t CustomCrop::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode CustomCrop::infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, NVCVRectI roi,
                            cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (!(input_format == kNHWC || input_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        printf("Invliad DataFormat both Input and Output must be kHWC or kHWC\n");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = cv::TensorDataAccessPitchImagePlanar::Create(inData);
    if (!inAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int batch    = inAccess->numSamples();
    int channels = inAccess->numChannels();
    int rows     = inAccess->numRows();
    int cols     = inAccess->numCols();

    if (channels > 4 || channels < 1)
    {
        LOG_ERROR("Invalid channel number ch = " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto outAccess = cv::TensorDataAccessPitchImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (roi.height > outAccess->size().h || roi.width > outAccess->size().w)
    {
        LOG_ERROR("ROI larger than dst buffer");
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int data_size = DataSize(GetLegacyDataType(inData.dtype()));
    int start_x   = roi.x;
    int start_y   = roi.y;
    int end_x     = start_x + roi.width - 1;
    int end_y     = start_y + roi.height - 1;
#ifdef CUDA_DEBUG_LOG
    printf("x %d, y %d, w %d, h %d\n", roi.x, roi.y, roi.width, roi.height);
#endif

    if (start_x < 0 || start_y < 0 || end_x >= cols || end_y >= rows)
    {
        printf("Invliad Roi range x %d, y %d, width %d, height %d\n", roi.x, roi.y, roi.width, roi.height);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const cv::TensorDataAccessPitchImagePlanar &inData,
                           const cv::TensorDataAccessPitchImagePlanar &outData, NVCVRectI roi, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {customCrop<uchar1>,  customCrop<uchar2>,  customCrop<uchar3>,  customCrop<uchar4>},
        {customCrop<ushort>, customCrop<ushort2>, customCrop<ushort3>, customCrop<ushort4>},
        {   customCrop<int>,    customCrop<int2>,    customCrop<int3>,    customCrop<int4>},
        {                 0,                   0,                   0,                   0},
        {customCrop<double>, customCrop<double2>, customCrop<double3>, customCrop<double4>}
    };

    funcs[data_size / 2][channels - 1](*inAccess, *outAccess, roi, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
