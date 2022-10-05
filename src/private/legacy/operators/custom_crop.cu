/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../CvCudaUtils.cuh"

#include <cstdio>

using namespace cuda_op;

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
void customCrop(const void *input, void *output, int start_x, int start_y, DataShape inputShape, DataShape outputShape,
                cudaStream_t stream)
{
    Ptr2dNHWC<T> src_ptr(inputShape.N, inputShape.H, inputShape.W, inputShape.C, (T *)input);
    Ptr2dNHWC<T> dst_ptr(outputShape.N, outputShape.H, outputShape.W, outputShape.C, (T *)output);

    dim3 block(16, 16);
    dim3 grid(divUp(outputShape.W, block.x), divUp(outputShape.H, block.y), outputShape.N);

    custom_crop_kernel<Ptr2dNHWC<T>><<<grid, block, 0, stream>>>(src_ptr, dst_ptr, start_x, start_y);
    checkKernelErrors();
}

namespace cuda_op {

size_t CustomCrop::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

int CustomCrop::infer(const void *const *inputs, void **outputs, void *workspace, Rect roi, const DataShape input_shape,
                      const DataFormat format, const DataType data_type, cudaStream_t stream)
{
    int batch    = input_shape.N;
    int channels = input_shape.C;
    int rows     = input_shape.H;
    int cols     = input_shape.W;

    if (!(format == kNHWC || format == kHWC))
    {
        printf("Invliad DataFormat %d\n", format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (channels > 4)
    {
        printf("Invalid channel number %d\n", channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    int data_size = DataSize(data_type);
    int start_x   = roi.x;
    int start_y   = roi.y;
    int end_x     = start_x + roi.width - 1;
    int end_y     = start_y + roi.height - 1;
#ifdef CUDA_DEBUG_LOG
    printf("x %d, y %d, w %d, h %d\n", roi.x, roi.y, roi.width, roi.height);
#endif
    cuda_op::DataShape output_shape(batch, channels, roi.height, roi.width);

    if (start_x < 0 || start_y < 0 || end_x >= cols || end_y >= rows)
    {
        printf("Invliad Roi range x %d, y %d, width %d, height %d\n", roi.x, roi.y, roi.width, roi.height);
        return ErrorCode::INVALID_PARAMETER;
    }

    typedef void (*func_t)(const void *input, void *output, int start_x, int start_y, DataShape inputShape,
                           DataShape outputShape, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {customCrop<uchar1>,  customCrop<uchar2>,  customCrop<uchar3>,  customCrop<uchar4>},
        {customCrop<ushort>, customCrop<ushort2>, customCrop<ushort3>, customCrop<ushort4>},
        {   customCrop<int>,    customCrop<int2>,    customCrop<int3>,    customCrop<int4>},
        {                 0,                   0,                   0,                   0},
        {customCrop<double>, customCrop<double2>, customCrop<double3>, customCrop<double4>}
    };

    funcs[data_size / 2][channels - 1](inputs[0], outputs[0], start_x, start_y, input_shape, output_shape, stream);

    return 0;
}

} // namespace cuda_op
