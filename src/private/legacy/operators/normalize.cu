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

#include <operators/OpNormalize.h> // for NVCV_OP_NORMALIZE_SCALE_IS_STDDEV, etc.

using namespace nv::cv::legacy::cuda_op;
using namespace nv::cv::legacy::helpers;

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename input_type, typename base_type, typename scale_type>
__global__ void normalizeKernel(const input_type src, const base_type base, const scale_type scale, input_type dst,
                                float global_scale, float global_shift)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= dst.cols || src_y >= dst.rows)
        return;

    const int base_x         = base.cols == 1 ? 0 : src_x;
    const int base_y         = base.rows == 1 ? 0 : src_y;
    const int base_batch_idx = base.batches == 1 ? 0 : batch_idx;

    const int scale_x         = scale.cols == 1 ? 0 : src_x;
    const int scale_y         = scale.rows == 1 ? 0 : src_y;
    const int scale_batch_idx = scale.batches == 1 ? 0 : batch_idx;

    typedef typename input_type::value_type input_value_type;

    *dst.ptr(batch_idx, src_y, src_x) = nv::cv::cuda::SaturateCast<nv::cv::cuda::BaseType<input_value_type>>(
        (*src.ptr(batch_idx, src_y, src_x) - *base.ptr(base_batch_idx, base_y, base_x))
            * (*scale.ptr(scale_batch_idx, scale_y, scale_x)) * global_scale
        + global_shift);
}

// (float3 - float3) * float3 / (float3 - float) * float3 / (float3 - float3) * float / (float3 - float) * float
template<typename input_type, typename base_type, typename scale_type>
__global__ void normalizeInvStdDevKernel(const input_type src, const base_type base, const scale_type scale,
                                         input_type dst, float global_scale, float global_shift, float epsilon)
{
    const int src_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int src_y     = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (src_x >= dst.cols || src_y >= dst.rows)
        return;

    const int base_x         = base.cols == 1 ? 0 : src_x;
    const int base_y         = base.rows == 1 ? 0 : src_y;
    const int base_batch_idx = base.batches == 1 ? 0 : batch_idx;

    const int scale_x         = scale.cols == 1 ? 0 : src_x;
    const int scale_y         = scale.rows == 1 ? 0 : src_y;
    const int scale_batch_idx = scale.batches == 1 ? 0 : batch_idx;

    typedef typename input_type::value_type input_value_type;
    typedef typename scale_type::value_type scale_value_type;

    scale_value_type s   = *scale.ptr(scale_batch_idx, scale_y, scale_x);
    scale_value_type x   = s * s + epsilon;
    scale_value_type mul = 1.0f / nv::cv::cuda::sqrt(x);

    *dst.ptr(batch_idx, src_y, src_x) = nv::cv::cuda::SaturateCast<nv::cv::cuda::BaseType<input_value_type>>(
        (*src.ptr(batch_idx, src_y, src_x) - *base.ptr(base_batch_idx, base_y, base_x)) * mul * global_scale
        + global_shift);
}

template<typename base_type, typename scale_type, typename PtrInput, typename PtrOutput>
void normalizeWrap(PtrInput src_ptr, PtrOutput dst_ptr, DataShape input_shape,
                   const nv::cv::ITensorDataPitchDevice &baseData, const nv::cv::ITensorDataPitchDevice &scaleData,
                   float global_scale, float shift, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(divUp(input_shape.W, block.x), divUp(input_shape.H, block.y), input_shape.N);

    Ptr2dNHWC<base_type>  base_ptr(baseData);
    Ptr2dNHWC<scale_type> scale_ptr(scaleData);

    normalizeKernel<<<grid, block, 0, stream>>>(src_ptr, base_ptr, scale_ptr, dst_ptr, global_scale, shift);
    checkKernelErrors();
}

template<typename base_type, typename scale_type, typename PtrInput, typename PtrOutput>
void normalizeInvStdDevWrap(PtrInput src_ptr, PtrOutput dst_ptr, DataShape input_shape,
                            const nv::cv::ITensorDataPitchDevice &baseData,
                            const nv::cv::ITensorDataPitchDevice &scaleData, float global_scale, float shift,
                            float epsilon, cudaStream_t stream)
{
    dim3 block(32, 8);
    dim3 grid(divUp(input_shape.W, block.x), divUp(input_shape.H, block.y), input_shape.N);

    Ptr2dNHWC<base_type>  base_ptr(baseData);
    Ptr2dNHWC<scale_type> scale_ptr(scaleData);

    normalizeInvStdDevKernel<<<grid, block, 0, stream>>>(src_ptr, base_ptr, scale_ptr, dst_ptr, global_scale, shift,
                                                         epsilon);
    checkKernelErrors();
}

template<typename input_type>
void normalize(const nv::cv::ITensorDataPitchDevice &inData, const nv::cv::ITensorDataPitchDevice &baseData,
               const nv::cv::ITensorDataPitchDevice &scaleData, const nv::cv::ITensorDataPitchDevice &outData,
               float global_scale, float shift, cudaStream_t stream)
{
    Ptr2dNHWC<input_type> src_ptr(inData);
    Ptr2dNHWC<input_type> dst_ptr(outData);

    DataShape input_shape = GetLegacyDataShape(inData.dims());

    using work_type = nv::cv::cuda::ConvertBaseTypeTo<float, input_type>;

    if (baseData.dims().c != 1 && scaleData.dims().c != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normalizeWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
    else if (baseData.dims().c != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normalizeWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
    else if (scaleData.dims().c != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normalizeWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normalizeWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale, shift,
                                             stream);
    }
}

template<typename input_type>
void normalizeInvStdDev(const nv::cv::ITensorDataPitchDevice &inData, const nv::cv::ITensorDataPitchDevice &baseData,
                        const nv::cv::ITensorDataPitchDevice &scaleData, const nv::cv::ITensorDataPitchDevice &outData,
                        float global_scale, float shift, float epsilon, cudaStream_t stream)
{
    Ptr2dNHWC<input_type> src_ptr(inData);
    Ptr2dNHWC<input_type> dst_ptr(outData);

    DataShape input_shape = GetLegacyDataShape(inData.dims());

    using work_type = nv::cv::cuda::ConvertBaseTypeTo<float, input_type>;

    if (baseData.dims().c != 1 && scaleData.dims().c != 1)
    {
        using base_type  = work_type;
        using scale_type = work_type;
        normalizeInvStdDevWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
    else if (baseData.dims().c != 1)
    {
        using base_type  = work_type;
        using scale_type = float;
        normalizeInvStdDevWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
    else if (scaleData.dims().c != 1)
    {
        using base_type  = float;
        using scale_type = work_type;
        normalizeInvStdDevWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
    else
    {
        using base_type  = float;
        using scale_type = float;
        normalizeInvStdDevWrap<base_type, scale_type>(src_ptr, dst_ptr, input_shape, baseData, scaleData, global_scale,
                                                      shift, epsilon, stream);
    }
}

namespace nv::cv::legacy::cuda_op {

void Normalize::checkParamShape(DataShape input_shape, DataShape param_shape)
{
    NVCV_ASSERT(param_shape.N == input_shape.N || param_shape.N == 1);
    NVCV_ASSERT(param_shape.C == input_shape.C || param_shape.C == 1);
    NVCV_ASSERT(param_shape.H == input_shape.H || param_shape.H == 1);
    NVCV_ASSERT(param_shape.W == input_shape.W || param_shape.W == 1);
}

size_t Normalize::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 0;
}

ErrorCode Normalize::infer(const nv::cv::ITensorDataPitchDevice &inData, const nv::cv::ITensorDataPitchDevice &baseData,
                           const nv::cv::ITensorDataPitchDevice &scaleData,
                           const nv::cv::ITensorDataPitchDevice &outData, const float global_scale, const float shift,
                           const float epsilon, const uint32_t flags, cudaStream_t stream)
{
    DataFormat format            = GetLegacyDataFormat(inData.layout(), inData.numImages());
    DataType   data_type         = GetLegacyDataType(inData.dtype());
    DataShape  input_shape       = GetLegacyDataShape(inData.dims());
    DataShape  base_param_shape  = GetLegacyDataShape(baseData.dims());
    DataShape  scale_param_shape = GetLegacyDataShape(scaleData.dims());

    int channels = input_shape.C;

    if (!(format == kNHWC || format == kHWC))
    {
        printf("Invalid DataFormat %d\n", format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    checkParamShape(input_shape, base_param_shape);
    checkParamShape(input_shape, scale_param_shape);

    typedef void (*normalize_t)(
        const nv::cv::ITensorDataPitchDevice &inData, const nv::cv::ITensorDataPitchDevice &baseData,
        const nv::cv::ITensorDataPitchDevice &scaleData, const nv::cv::ITensorDataPitchDevice &outData,
        float global_scale, float shift, cudaStream_t stream);

    typedef void (*normalizeInvStdDev_t)(
        const nv::cv::ITensorDataPitchDevice &inData, const nv::cv::ITensorDataPitchDevice &baseData,
        const nv::cv::ITensorDataPitchDevice &scaleData, const nv::cv::ITensorDataPitchDevice &outData,
        float global_scale, float shift, float epsilon, cudaStream_t stream);

    static const normalize_t funcs_normalize[6][4] = {
        { normalize<uchar>,  0 /*normalize<uchar2>*/,  normalize<uchar3>,  normalize<uchar4>},
        { normalize<schar>,   0 /*normalize<char2>*/,   normalize<char3>,   normalize<char4>},
        {normalize<ushort>, 0 /*normalize<ushort2>*/, normalize<ushort3>, normalize<ushort4>},
        { normalize<short>,  0 /*normalize<short2>*/,  normalize<short3>,  normalize<short4>},
        {   normalize<int>,    0 /*normalize<int2>*/,    normalize<int3>,    normalize<int4>},
        { normalize<float>,  0 /*normalize<float2>*/,  normalize<float3>,  normalize<float4>}
    };

    static const normalizeInvStdDev_t funcs_normalize_stddev[6][4] = {
        { normalizeInvStdDev<uchar>,  0 /*normalizeInvStdDev<uchar2>*/,  normalizeInvStdDev<uchar3>,
         normalizeInvStdDev<uchar4>                                                                                          },
        { normalizeInvStdDev<schar>,   0 /*normalizeInvStdDev<char2>*/,   normalizeInvStdDev<char3>,
         normalizeInvStdDev<char4>                                                                                           },
        {normalizeInvStdDev<ushort>, 0 /*normalizeInvStdDev<ushort2>*/, normalizeInvStdDev<ushort3>,
         normalizeInvStdDev<ushort4>                                                                                         },
        { normalizeInvStdDev<short>,  0 /*normalizeInvStdDev<short2>*/,  normalizeInvStdDev<short3>,
         normalizeInvStdDev<short4>                                                                                          },
        {   normalizeInvStdDev<int>,    0 /*normalizeInvStdDev<int2>*/,    normalizeInvStdDev<int3>, normalizeInvStdDev<int4>},
        { normalizeInvStdDev<float>,  0 /*normalizeInvStdDev<float2>*/,  normalizeInvStdDev<float3>,
         normalizeInvStdDev<float4>                                                                                          }
    };

    if (flags & NVCV_OP_NORMALIZE_SCALE_IS_STDDEV)
    {
        funcs_normalize_stddev[data_type][channels - 1](inData, baseData, scaleData, outData, global_scale, shift,
                                                        epsilon, stream);
    }
    else
    {
        funcs_normalize[data_type][channels - 1](inData, baseData, scaleData, outData, global_scale, shift, stream);
    }

    return SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
