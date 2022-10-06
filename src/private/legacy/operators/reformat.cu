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

#include <cassert>
#include <cstdio>

using namespace nv::cv::legacy::cuda_op;
using namespace nv::cv::legacy::helpers;

namespace nvcv = nv::cv;

template<typename Ptr2DSrc, typename Ptr2DDst>
__global__ void transformFormat(const Ptr2DSrc src, Ptr2DDst dst)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= dst.cols || src_y >= dst.rows)
        return;

    for (int c = 0; c < dst.ch; c++)
    {
        *dst.ptr(batch_idx, src_y, src_x, c) = *src.ptr(batch_idx, src_y, src_x, c);
    }
}

template<typename data_type> // uchar float
void nhwc_to_nchw(const nvcv::ITensorDataPitchDevice &inData, const nvcv::ITensorDataPitchDevice &outData,
                  cudaStream_t stream)
{
    const int cols       = inData.dims().w;
    const int rows       = inData.dims().h;
    const int batch_size = 1;

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), batch_size);

    Ptr2dNHWC<data_type> src_ptr(inData);
    Ptr2dNCHW<data_type> dst_ptr(outData);

    transformFormat<<<grid, block, 0, stream>>>(src_ptr, dst_ptr);

    checkKernelErrors();
}

template<typename data_type> // uchar float
void nchw_to_nhwc(const nvcv::ITensorDataPitchDevice &inData, const nvcv::ITensorDataPitchDevice &outData,
                  cudaStream_t stream)
{
    const int cols       = inData.dims().w;
    const int rows       = inData.dims().h;
    const int batch_size = 1;

    dim3 block(32, 8);
    dim3 grid(divUp(cols, block.x), divUp(rows, block.y), batch_size);

    Ptr2dNCHW<data_type> src_ptr(inData);
    Ptr2dNHWC<data_type> dst_ptr(outData);

    transformFormat<<<grid, block, 0, stream>>>(src_ptr, dst_ptr);
    checkKernelErrors();

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
}

namespace nv::cv::legacy::cuda_op {

void Reformat::checkDataFormat(DataFormat format)
{
    assert(format == kNHWC || format == kHWC || format == kNCHW || format == kCHW);
}

size_t Reformat::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode Reformat::infer(const nvcv::ITensorDataPitchDevice &inData, const nvcv::ITensorDataPitchDevice &outData,
                          cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (inData.format() == outData.format())
    {
#ifdef CUDA_DEBUG_LOG
        printf("input_format == output_format, copy outputs from inputs\n");
#endif

        for (uint32_t i = 0; i < inData.numImages(); ++i)
        {
            for (uint32_t p = 0; p < inData.numPlanes(); ++p)
            {
                checkCudaErrors(cudaMemcpy2DAsync(outData.imgPlaneBuffer(i, p), outData.rowPitchBytes(),
                                                  inData.imgPlaneBuffer(i, p), inData.rowPitchBytes(),
                                                  inData.dims().w * inData.colPitchBytes(), inData.dims().h,
                                                  cudaMemcpyDeviceToDevice, stream));
            }
        }
        return SUCCESS;
    }

    cuda_op::DataType data_type = GetLegacyDataType(inData.format());

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F || data_type == kCV_64F))
    {
        printf("Invalid DataType %d\n", data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    typedef void (*transform_t)(const nvcv::ITensorDataPitchDevice &input, const nvcv::ITensorDataPitchDevice &output,
                                cudaStream_t stream);

    checkDataFormat(input_format);
    checkDataFormat(output_format);

    if ((input_format == kNHWC || input_format == kHWC) && (output_format == kNCHW || output_format == kCHW))
    {
        static const transform_t funcs[7]
            = {nhwc_to_nchw<uchar>, nhwc_to_nchw<schar>, nhwc_to_nchw<ushort>, nhwc_to_nchw<short>,
               nhwc_to_nchw<int>,   nhwc_to_nchw<float>, nhwc_to_nchw<double>};

        transform_t func = funcs[data_type];
        func(inData, outData, stream);
        return SUCCESS;
    }

    if ((input_format == kNCHW || input_format == kCHW) && (output_format == kNHWC || output_format == kHWC))
    {
        static const transform_t funcs[7]
            = {nchw_to_nhwc<uchar>, nchw_to_nhwc<schar>, nchw_to_nhwc<ushort>, nchw_to_nhwc<short>,
               nchw_to_nhwc<int>,   nchw_to_nhwc<float>, nchw_to_nhwc<double>};
        transform_t func = funcs[data_type];
        func(inData, outData, stream);
        return SUCCESS;
    }

    return SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
