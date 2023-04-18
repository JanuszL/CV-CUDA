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
static __host__ __device__ unsigned char u8cast(_T value) {
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

// inbox_single_pixel:
// check if given coordinate is in box
//      a --- d
//      |     |
//      b --- c
static __device__ __forceinline__ bool inbox_single_pixel(
    float ix, float iy, float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy) {
    return  ((bx-ax) * (iy - ay) - (by-ay) * (ix-ax)) < 0 &&
            ((cx-bx) * (iy - by) - (cy-by) * (ix-bx)) < 0 &&
            ((dx-cx) * (iy - cy) - (dy-cy) * (ix-cx)) < 0 &&
            ((ax-dx) * (iy - dy) - (ay-dy) * (ix-dx)) < 0;
}

static __device__ void blend_single_color(uchar4& color, unsigned char& c0, unsigned char& c1, unsigned char& c2, unsigned char a) {
    int foreground_alpha = a;
    int background_alpha = color.w;
    int blend_alpha      = ((background_alpha * (255 - foreground_alpha))>> 8) + foreground_alpha;
    color.x = u8cast((((color.x * background_alpha * (255 - foreground_alpha))>>8) + (c0 * foreground_alpha)) / blend_alpha);
    color.y = u8cast((((color.y * background_alpha * (255 - foreground_alpha))>>8) + (c1 * foreground_alpha)) / blend_alpha);
    color.z = u8cast((((color.z * background_alpha * (255 - foreground_alpha))>>8) + (c2 * foreground_alpha)) / blend_alpha);
    color.w = blend_alpha;
}

// render_rectangle_fill:
// render filled rectangle with border msaa4x interpolation off
static __device__ void render_rectangle_fill(int ix, int iy, RectangleCommand* p, uchar4 color[4]) {
    if (inbox_single_pixel(ix, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)) {
        blend_single_color(color[0], p->c0, p->c1, p->c2, p->c3);
    }
    if (inbox_single_pixel(ix+1, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)) {
        blend_single_color(color[1], p->c0, p->c1, p->c2, p->c3);
    }
    if (inbox_single_pixel(ix, iy+1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)) {
        blend_single_color(color[2], p->c0, p->c1, p->c2, p->c3);
    }
    if (inbox_single_pixel(ix+1, iy+1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)) {
        blend_single_color(color[3], p->c0, p->c1, p->c2, p->c3);
    }
}

// render_rectangle_border:
// render hollow rectangle with border msaa4x interpolation off
static __device__ void render_rectangle_border(int ix, int iy, RectangleCommand* p, uchar4 color[4]) {
    if (!inbox_single_pixel(ix, iy, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2) &&
        inbox_single_pixel(ix, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)
    ) {
        blend_single_color(color[0], p->c0, p->c1, p->c2, p->c3);
    }
    if (!inbox_single_pixel(ix+1, iy, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2) &&
        inbox_single_pixel(ix+1, iy, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)
    ) {
        blend_single_color(color[1], p->c0, p->c1, p->c2, p->c3);
    }
    if (!inbox_single_pixel(ix, iy+1, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2) &&
        inbox_single_pixel(ix, iy+1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)
    ) {
        blend_single_color(color[2], p->c0, p->c1, p->c2, p->c3);
    }
    if (!inbox_single_pixel(ix+1, iy+1, p->ax2, p->ay2, p->bx2, p->by2, p->cx2, p->cy2, p->dx2, p->dy2) &&
        inbox_single_pixel(ix+1, iy+1, p->ax1, p->ay1, p->bx1, p->by1, p->cx1, p->cy1, p->dx1, p->dy1)
    ) {
        blend_single_color(color[3], p->c0, p->c1, p->c2, p->c3);
    }
}

static __device__ void do_rectangle_woMSAA(RectangleCommand* cmd, int ix, int iy, uchar4 context_color[4]) {
    if (cmd->thickness == -1) {
        render_rectangle_fill(ix, iy, cmd, context_color);
    } else {
        render_rectangle_border(ix, iy, cmd, context_color);
    }
}

template<class SrcWrapper, class DstWrapper, typename T = typename DstWrapper::ValueType>
static __device__ void blending_pixel(
    SrcWrapper src, DstWrapper dst,
    int x, int y, uchar4 plot_colors[4]
)
{
    const int batch_idx = get_batch_idx();

    for (int i = 0; i < 2; ++i) {
        T* in = src.ptr(batch_idx, y + i, x, 0);
        T* out = dst.ptr(batch_idx, y + i, x, 0);
        for (int j = 0; j < 2; ++j, in += 4, out += 4) {
            uchar4& rcolor   = plot_colors[i * 2 + j];
            int foreground_alpha = rcolor.w;
            int background_alpha = in[3];
            int blend_alpha      = ((background_alpha * (255 - foreground_alpha)) >> 8) + foreground_alpha;
            out[0] = u8cast((((in[0] * background_alpha * (255 - foreground_alpha))>>8) + (rcolor.x * foreground_alpha)) / blend_alpha);
            out[1] = u8cast((((in[1] * background_alpha * (255 - foreground_alpha))>>8) + (rcolor.y * foreground_alpha)) / blend_alpha);
            out[2] = u8cast((((in[2] * background_alpha * (255 - foreground_alpha))>>8) + (rcolor.z * foreground_alpha)) / blend_alpha);
            out[3] = blend_alpha;
        }
    }
}

template<class SrcWrapper, class DstWrapper>
static __global__ void render_bndbox_rgba_womsaa_kernel(
    SrcWrapper src, DstWrapper dst, int bx, int by,
    const unsigned char* commands, const int* command_offsets, int num_command,
    int width, int height
) {
    int ix = ((blockDim.x * blockIdx.x + threadIdx.x) << 1) + bx;
    int iy = ((blockDim.y * blockIdx.y + threadIdx.y) << 1) + by;
    if (ix < 0 || iy < 0 || ix >= width - 1 || iy >= height - 1)
        return;

    uchar4 context_color[4] = {0};

    for (int i = 0; i < num_command; ++i) {
        RectangleCommand* pcommand = (RectangleCommand*)(commands + command_offsets[i]);
        if(pcommand->batch_index != get_batch_idx()) continue;
        do_rectangle_woMSAA(pcommand, ix, iy, context_color);
    }

    if (context_color[0].w == 0 && context_color[1].w == 0 && context_color[2].w == 0 && context_color[3].w == 0)
        return;

    blending_pixel(src, dst, ix, iy, context_color);
}

static void cuosd_apply(
    cuOSDContext_t context, int width, int height, cudaStream_t stream
) 
{
    context->bounding_left   = width;
    context->bounding_top    = height;
    context->bounding_right  = 0;
    context->bounding_bottom = 0;

    size_t byte_of_commands = 0;
    std::vector<unsigned int> cmd_offset(context->commands.size());
    for (int i = 0; i < (int)context->commands.size(); ++i) {
        auto& cmd = context->commands[i];
        cmd_offset[i] = byte_of_commands;

        context->bounding_left   = min(context->bounding_left,   cmd->bounding_left);
        context->bounding_top    = min(context->bounding_top,    cmd->bounding_top);
        context->bounding_right  = max(context->bounding_right,  cmd->bounding_right);
        context->bounding_bottom = max(context->bounding_bottom, cmd->bounding_bottom);

        byte_of_commands += sizeof(RectangleCommand);
    }

    if (context->gpu_commands        == nullptr) context->gpu_commands.reset(new Memory<unsigned char>());
    if (context->gpu_commands_offset == nullptr) context->gpu_commands_offset.reset(new Memory<int>());

    context->gpu_commands->alloc_or_resize_to(byte_of_commands);
    context->gpu_commands_offset->alloc_or_resize_to(cmd_offset.size());
    memcpy(context->gpu_commands_offset->host(), cmd_offset.data(), sizeof(int) * cmd_offset.size());

    for (int i = 0; i < (int)context->commands.size(); ++i) {
        auto& cmd       = context->commands[i];
        unsigned char* pg_cmd = context->gpu_commands->host() + cmd_offset[i];
        memcpy(pg_cmd, cmd.get(), sizeof(RectangleCommand));
    }
    context->gpu_commands->copy_host_to_device(stream);
    context->gpu_commands_offset->copy_host_to_device(stream);
}


inline ErrorCode ApplyBndBox_RGBA(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
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

    dim3 blockSize(16, 8);
    dim3 gridSize(divUp(int((inputShape.W + 1) / 2), (int)blockSize.x), divUp(int((inputShape.H + 1) / 2), (int)blockSize.y), inputShape.N);

    auto src = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(inData);
    auto dst = nvcv::cuda::CreateTensorWrapNHWC<uint8_t>(outData);

    // allocate command buffer;
    cuosd_apply(context, inputShape.W, inputShape.H, stream);

    render_bndbox_rgba_womsaa_kernel<<<gridSize, blockSize, 0, stream>>>(
        src, dst, 0, 0,
        context->gpu_commands ? context->gpu_commands->device() : nullptr,
        context->gpu_commands_offset ? context->gpu_commands_offset->device() : nullptr,
        context->commands.size(),
        inputShape.W, inputShape.H);
    checkKernelErrors();

    return ErrorCode::SUCCESS;
}

static ErrorCode cuosd_draw_rectangle(cuOSDContext_t context, int width, int height, NVCVBndBoxesI bboxes){
    for (int n = 0; n < bboxes.batch; n++)
    {
        auto numBoxes = bboxes.numBoxes[n];

        for (int i = 0; i < numBoxes; i++)
        {
            auto bbox = bboxes.boxes[i];
            int left    = max(min(bbox.rect.x, width - 1),    0);
            int top     = max(min(bbox.rect.y, height - 1),   0);
            int right   = max(min(left + bbox.rect.width - 1, width - 1),  0);
            int bottom  = max(min(top + bbox.rect.height - 1, height - 1), 0);

            if (left == right || top == bottom || bbox.rect.width <= 0 || bbox.rect.height <= 0)
            {
                // LOG_INFO("Skipped bbox rect(" << bbox.rect.x << ", " << bbox.rect.y << ", "<< bbox.rect.width << ", " << bbox.rect.height
                //         << ") in image(" << width << ", " << height << ")");
                continue;
            }

            if (bbox.borderColor.a == 0) continue;
            if (bbox.fillColor.a || bbox.thickness == -1) {
                if (bbox.thickness == -1) {
                    bbox.fillColor = bbox.borderColor;
                }

                auto cmd = std::make_shared<RectangleCommand>();
                cmd->batch_index = n;
                cmd->thickness = -1;
                cmd->interpolation = false;
                cmd->c0 = bbox.fillColor.r; cmd->c1 = bbox.fillColor.g; cmd->c2 = bbox.fillColor.b; cmd->c3 = bbox.fillColor.a;

                // a   d
                // b   c
                cmd->ax1 = left; cmd->ay1 = top; cmd->dx1 = right; cmd->dy1 = top; cmd->cx1 = right; cmd->cy1 = bottom; cmd->bx1 = left; cmd->by1 = bottom;
                cmd->bounding_left  = left; cmd->bounding_right = right; cmd->bounding_top   = top; cmd->bounding_bottom = bottom;
                context->commands.emplace_back(cmd);
            }
            if (bbox.thickness == -1) continue;

            auto cmd = std::make_shared<RectangleCommand>();
            cmd->batch_index = n;
            cmd->thickness = bbox.thickness;
            cmd->interpolation = false;
            cmd->c0 = bbox.borderColor.r; cmd->c1 = bbox.borderColor.g; cmd->c2 = bbox.borderColor.b; cmd->c3 = bbox.borderColor.a;

            float half_thickness = bbox.thickness / 2.0f;
            cmd->ax2 = left + half_thickness;
            cmd->ay2 = top  + half_thickness;
            cmd->dx2 = right - half_thickness;
            cmd->dy2 = top + half_thickness;
            cmd->cx2 = right - half_thickness;
            cmd->cy2 = bottom - half_thickness;
            cmd->bx2 = left + half_thickness;
            cmd->by2 = bottom - half_thickness;

            // a   d
            // b   c
            cmd->ax1 = left - half_thickness;
            cmd->ay1 = top  - half_thickness;
            cmd->dx1 = right + half_thickness;
            cmd->dy1 = top - half_thickness;
            cmd->cx1 = right + half_thickness;
            cmd->cy1 = bottom + half_thickness;
            cmd->bx1 = left - half_thickness;
            cmd->by1 = bottom + half_thickness;

            int int_half = ceil(half_thickness);
            cmd->bounding_left  = left - int_half;
            cmd->bounding_right = right + int_half;
            cmd->bounding_top   = top - int_half;
            cmd->bounding_bottom = bottom + int_half;
            context->commands.emplace_back(cmd);
        }

        bboxes.boxes = (NVCVBndBoxI *)((unsigned char *)bboxes.boxes + numBoxes * sizeof(NVCVBndBoxI));
    }
    return ErrorCode::SUCCESS;
}

BndBox::BndBox(DataShape max_input_shape, DataShape max_output_shape)
    : CudaBaseOp(max_input_shape, max_output_shape)
{
    m_context = new cuOSDContext();
}

BndBox::~BndBox(){
    if (m_context) {
        m_context->commands.clear();
        cuOSDContext* p = (cuOSDContext*)m_context;
        delete p;
    }
}

size_t BndBox::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode BndBox::infer(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                        NVCVBndBoxesI bboxes, cudaStream_t stream)
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

    if (bboxes.batch != batch)
    {
        LOG_ERROR("Invalid bboxes batch = " << bboxes.batch);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    auto outAccess = nvcv::TensorDataAccessStridedImagePlanar::Create(outData);
    if (!outAccess)
    {
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    auto ret = cuosd_draw_rectangle(m_context, cols, rows, bboxes);
    if (ret != ErrorCode::SUCCESS) {
        return ret;
    }

    typedef ErrorCode (*func_t)(const nvcv::TensorDataStridedCuda &inData, const nvcv::TensorDataStridedCuda &outData,
                                cuOSDContext_t context, cudaStream_t stream);

    static const func_t funcs[] = {
        ApplyBndBox_RGBA,
    };

    funcs[0](
        inData, outData, m_context, stream
    );

    return ErrorCode::SUCCESS;
}

} // namespace nvcv::legacy::cuda_op
