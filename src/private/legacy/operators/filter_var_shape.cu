/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

using namespace nv::cv::legacy::cuda_op;
using namespace nv::cv::legacy::helpers;

namespace nv::cv::legacy::cuda_op {

template<typename D, typename BrdRd>
__global__ void filter2D(const BrdRd src, Ptr2dVarShapeNHWC<D> dst, Ptr2dVarShapeNHWC<float> kernel,
                         cuda::Tensor1DWrap<int2> kernelAnchor)
{
    using work_type = cuda::ConvertBaseTypeTo<float, D>;
    work_type res   = cuda::SetAll<work_type>(0);

    const int x         = blockIdx.x * blockDim.x + threadIdx.x;
    const int y         = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = get_batch_idx();

    if (x >= dst.at_cols(batch_idx) || y >= dst.at_rows(batch_idx))
        return;

    int2 anchor = *kernelAnchor.ptr(batch_idx);

    int2 kernelSize{kernel.at_cols(batch_idx), kernel.at_rows(batch_idx)};

    if (anchor.x < 0)
        anchor.x = kernelSize.x / 2;

    if (anchor.y < 0)
        anchor.y = kernelSize.y / 2;

    for (int i = 0; i < kernelSize.y; ++i)
    {
        for (int j = 0; j < kernelSize.x; ++j)
        {
            res = res + (src(batch_idx, y - anchor.y + i, x - anchor.x + j)) * (*kernel.ptr(batch_idx, i, j));
        }
    }

    *dst.ptr(batch_idx, y, x) = cuda::SaturateCast<cuda::BaseType<D>>(res);
}

template<typename D, template<typename> class Brd>
void Filter2DCaller(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const IImageBatchVarShapeDataPitchDevice &kernelData,
                    const ITensorDataPitchDevice &kernelAnchorData, float borderValue, cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<D> src(inData);
    Ptr2dVarShapeNHWC<D> dst(outData);

    Ptr2dVarShapeNHWC<float> kernel(kernelData);
    cuda::Tensor1DWrap<int2> kernelAnchor(kernelAnchorData);

    using work_type = cuda::ConvertBaseTypeTo<float, D>;

    dim3 block(16, 16);
    dim3 grid(divUp(inData.maxSize().w, block.x), divUp(inData.maxSize().h, block.y), outData.numImages());

    Brd<work_type>                                     brd(0, 0, cuda::SetAll<work_type>(borderValue));
    BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>> brdSrc(src, brd);

#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif

    filter2D<D, BorderReader<Ptr2dVarShapeNHWC<D>, Brd<work_type>>>
        <<<grid, block, 0, stream>>>(brdSrc, dst, kernel, kernelAnchor);
    checkKernelErrors();
#ifdef CUDA_DEBUG_LOG
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());
#endif
}

template<typename D>
void Filter2D(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
              const IImageBatchVarShapeDataPitchDevice &kernelData, const ITensorDataPitchDevice &kernelAnchorData,
              NVCVBorderType borderMode, float borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const IImageBatchVarShapeDataPitchDevice &inData,
                           const IImageBatchVarShapeDataPitchDevice &outData,
                           const IImageBatchVarShapeDataPitchDevice &kernelData,
                           const ITensorDataPitchDevice &kernelAnchorData, float borderValue, cudaStream_t stream);

    static const func_t funcs[]
        = {Filter2DCaller<D, BrdConstant>, Filter2DCaller<D, BrdReplicate>, Filter2DCaller<D, BrdReflect>,
           Filter2DCaller<D, BrdWrap>, Filter2DCaller<D, BrdReflect101>};

    funcs[borderMode](inData, outData, kernelData, kernelAnchorData, borderValue, stream);
}

// Conv2DVarShape --------------------------------------------------------------

ErrorCode Conv2DVarShape::infer(const IImageBatchVarShapeDataPitchDevice &inData,
                                const IImageBatchVarShapeDataPitchDevice &outData,
                                const IImageBatchVarShapeDataPitchDevice &kernelData,
                                const ITensorDataPitchDevice &kernelAnchorData, NVCVBorderType borderMode,
                                cudaStream_t stream)
{
    DataFormat input_format  = helpers::GetLegacyDataFormat(inData);
    DataFormat output_format = helpers::GetLegacyDataFormat(outData);
    if (input_format != output_format)
    {
        LOG_ERROR("Invalid DataFormat between input (" << input_format << ") and output (" << output_format << ")");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataFormat format = input_format;

    if (!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    if (!(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
          || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT || borderMode == NVCV_BORDER_WRAP))
    {
        LOG_ERROR("Invalid borderMode " << borderMode);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (!inData.uniqueFormat())
    {
        printf("Images in the input batch must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_16U || data_type == kCV_16S || data_type == kCV_32S
          || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    const int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    float borderValue = .0f;

    typedef void (*filter2D_t)(
        const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
        const IImageBatchVarShapeDataPitchDevice &kernelData, const ITensorDataPitchDevice &kernelAnchorData,
        NVCVBorderType borderMode, float borderValue, cudaStream_t stream);

    static const filter2D_t funcs[6][4] = {
        { Filter2D<uchar>, 0,  Filter2D<uchar3>,  Filter2D<uchar4>},
        {               0, 0,                 0,                 0},
        {Filter2D<ushort>, 0, Filter2D<ushort3>, Filter2D<ushort4>},
        { Filter2D<short>, 0,  Filter2D<short3>,  Filter2D<short4>},
        {   Filter2D<int>, 0,    Filter2D<int3>,    Filter2D<int4>},
        { Filter2D<float>, 0,  Filter2D<float3>,  Filter2D<float4>},
    };

    const filter2D_t func = funcs[data_type][channels - 1];

    NVCV_ASSERT(func != 0);

    func(inData, outData, kernelData, kernelAnchorData, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op