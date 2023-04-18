/* Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CvCudaLegacy.h"
#include "CvCudaLegacyHelpers.hpp"

#include "CvCudaUtils.cuh"

#include <nvcv/IImage.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/TensorData.hpp>

#include <cstdio>

using namespace nvcv::legacy::cuda_op;
using namespace nvcv::legacy::helpers;
using namespace nvcv::cuda::osd;

namespace nvcv::legacy::cuda_op {

template<typename _T>
static __forceinline__ __device__ _T limit(_T value, _T low, _T high){
    return value < low ? low : (value > high ? high : value);
}

template<class SrcWrapper, class DstWrapper>
static __global__ void render_blur_rgba_kernel(
    SrcWrapper src, DstWrapper dst,
    const BoxBlurCommand* commands, int num_command,
    int image_width, int image_height
) {
    if (blockIdx.x >= num_command) return;

    __shared__ uchar3 crop[32][32];
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    const BoxBlurCommand& box = commands[blockIdx.x];

    int boxwidth  = box.bounding_right  - box.bounding_left;
    int boxheight = box.bounding_bottom - box.bounding_top;
    int sx = limit((int)(ix / 32.0f * (float)boxwidth + 0.5f + box.bounding_left), 0, image_width);
    int sy = limit((int)(iy / 32.0f * (float)boxheight + 0.5f + box.bounding_top), 0, image_height);

    const int batch_idx = get_batch_idx();
    crop[iy][ix] = *(uchar3*)(src.ptr(batch_idx, sy, sx, 0));
    __syncthreads();

    uint3 color = make_uint3(0, 0, 0);
    int n = 0;
    for(int i = -box.kernel_size / 2; i <= box.kernel_size / 2; ++i){
        for(int j = -box.kernel_size / 2; j <= box.kernel_size / 2; ++j){
            int u = i + iy;
            int v = j + ix;
            if(u >= 0 && u < 32 && v >= 0 && v < 32){
                auto& c = crop[u][v];
                color.x += c.x;
                color.y += c.y;
                color.z += c.z;
                n++;
            }
        }
    }
    __syncthreads();
    crop[iy][ix] = make_uchar3(color.x / n, color.y / n, color.z / n);
    __syncthreads();

    int gap_width  = (boxwidth  + 31) / 32;
    int gap_height = (boxheight + 31) / 32;
    for(int i = 0; i < gap_height; ++i){
        for(int j = 0; j < gap_width; ++j){
            int fx = ix * gap_width + j + box.bounding_left;
            int fy = iy * gap_height + i + box.bounding_top;
            if(fx >= 0 && fx < image_width && fy >= 0 && fy < image_height){
                int sx = (ix * gap_width + j) / (float)boxwidth * 32;
                int sy = (iy * gap_height + i) / (float)boxheight * 32;
                if(sx < 32 && sy < 32){
                    auto& pix = crop[sy][sx];
                    *(uchar4*)(dst.ptr(batch_idx, fy, fx, 0)) = make_uchar4(pix.x, pix.y, pix.z, 255);
                }
            }
        }
    }
}

static void cuosd_apply(
    cuOSDContext_t context, cudaStream_t stream
) 
{
    if(!context->blur_commands.empty()){
        if (context->gpu_blur_commands == nullptr) context->gpu_blur_commands.reset(new Memory<BoxBlurCommand>());
        context->gpu_blur_commands->alloc_or_resize_to(context->blur_commands.size());
        for (int i = 0; i < (int)context->blur_commands.size(); ++i) {
            auto& cmd = context->blur_commands[i];
            memcpy((void*)(context->gpu_blur_commands->host() + i), (void*)cmd.get(), sizeof(BoxBlurCommand));
        }
        context->gpu_blur_commands->copy_host_to_device(stream);
    }
}


inline ErrorCode ApplyBoxBlur_RGBA(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                   cuOSDContext_t context, cudaStream_t stream)
{
    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    cuda_op::DataType  inDataType = helpers::GetLegacyDataType(inData.dtype());
    cuda_op::DataShape inputShape = helpers::GetLegacyDataShape(inAccess->infoShape());

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  outDataType = helpers::GetLegacyDataType(outData.dtype());
    cuda_op::DataShape outputShape = helpers::GetLegacyDataShape(outAccess->infoShape());

    if (outDataType != inDataType)
    {
        LOG_ERROR("Unsupported input/output DataType " << inDataType << "/" << outDataType);
        return ErrorCode::INVALID_DATA_TYPE;
    }
    if (outputShape.H != inputShape.H || outputShape.W != inputShape.W || outputShape.N != inputShape.N
        || outputShape.C != inputShape.C)
    {
        LOG_ERROR("Invalid output shape " << outputShape);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    // allocate command buffer;
    cuosd_apply(context, stream);

    dim3 blockSize(32, 32);
    dim3 gridSize(context->blur_commands.size());

    auto src = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(inData);
    auto dst = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(outData);

    render_blur_rgba_kernel<<<gridSize, blockSize, 0, stream>>>(
        src, dst,
        context->gpu_blur_commands ? context->gpu_blur_commands->device() : nullptr,
        context->blur_commands.size(),
        inputShape.W, inputShape.H);
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_boxblur(cuOSDContext_t context, NVCVBlurBoxesI bboxes){

    for (int i = 0; i < bboxes.box_num; i++) {
        auto bbox   = bboxes.boxes[i];

        int left    = bbox.rect.x;
        int top     = bbox.rect.y;
        int right   = left + bbox.rect.width - 1;
        int bottom  = top + bbox.rect.height - 1;

        if (bbox.rect.width < 3 || bbox.rect.height < 3 || bbox.kernelSize < 1)
        {
            LOG_ERROR("This operation will be ignored because the region of interest is too small, or the kernel is too small."
                      << bbox.rect.width << " " << bbox.rect.height << " " << bbox.kernelSize);
            return ErrorCode::INVALID_PARAMETER;
        }

        auto cmd = std::make_shared<BoxBlurCommand>();
        cmd->kernel_size = bbox.kernelSize;
        cmd->bounding_left    = left;
        cmd->bounding_right   = right;
        cmd->bounding_top     = top;
        cmd->bounding_bottom  = bottom;
        context->blur_commands.emplace_back(cmd);
    }
    return ErrorCode::SUCCESS;
}

BoxBlur::BoxBlur(DataShape max_input_shape, DataShape max_output_shape)
    : CudaBaseOp(max_input_shape, max_output_shape)
{
    m_context = new cuOSDContext();
}

BoxBlur::~BoxBlur(){
    if (m_context) {
        m_context->blur_commands.clear();
        cuOSDContext* p = (cuOSDContext*)m_context;
        delete p;
    }
}

size_t BoxBlur::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode BoxBlur::infer(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        NVCVBlurBoxesI bboxes, cudaStream_t stream)
{
    cuda_op::DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    cuda_op::DataFormat output_format = GetLegacyDataFormat(outData.layout());

    if (!(input_format == kNHWC || input_format == kHWC) || !(output_format == kNHWC || output_format == kHWC))
    {
        LOG_ERROR("Invliad DataFormat both Input and Output must be kNHWC or kHWC");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (inData.dtype() != outData.dtype())
    {
        LOG_ERROR("Input and Output formats must be same input format =" << inData.dtype()
                                                                         << " output format = " << outData.dtype());
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto inAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(inData);
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

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (bboxes.box_num <= 0)
    {
        LOG_ERROR("Invalid bbox num = " << bboxes.box_num);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto ret = cuosd_draw_boxblur(m_context, bboxes);
    if (ret != ErrorCode::SUCCESS) {
        return ret;
    }

    typedef ErrorCode (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                cuOSDContext_t context, cudaStream_t stream);

    static const func_t funcs[] = {
        ApplyBoxBlur_RGBA,
    };

    funcs[0](
        inData, outData, m_context, stream
    );

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
