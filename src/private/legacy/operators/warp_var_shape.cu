/* Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Copyright (C) 2021-2022, Bytedance Inc. All rights reserved.
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

#define BLOCK 32

using namespace nv::cv::legacy::cuda_op;
using namespace nv::cv::legacy::helpers;
using namespace nv::cv::cuda;

namespace nv::cv::legacy::cuda_op {

__global__ void inverseMatWarpPerspective(const int numImages, const cuda::Tensor2DWrap<float> in,
                                          cuda::Tensor2DWrap<float> out)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= numImages)
    {
        return;
    }

    cuda::math::Matrix<float, 3, 3> transMatrix;
    transMatrix[0][0] = (float)(*in.ptr(index, 0));
    transMatrix[0][1] = (float)(*in.ptr(index, 1));
    transMatrix[0][2] = (float)(*in.ptr(index, 2));
    transMatrix[1][0] = (float)(*in.ptr(index, 3));
    transMatrix[1][1] = (float)(*in.ptr(index, 4));
    transMatrix[1][2] = (float)(*in.ptr(index, 5));
    transMatrix[2][0] = (float)(*in.ptr(index, 6));
    transMatrix[2][1] = (float)(*in.ptr(index, 7));
    transMatrix[2][2] = (float)(*in.ptr(index, 8));

    cuda::math::inv_inplace(transMatrix);

    *out.ptr(index, 0) = transMatrix[0][0];
    *out.ptr(index, 1) = transMatrix[0][1];
    *out.ptr(index, 2) = transMatrix[0][2];
    *out.ptr(index, 3) = transMatrix[1][0];
    *out.ptr(index, 4) = transMatrix[1][1];
    *out.ptr(index, 5) = transMatrix[1][2];
    *out.ptr(index, 6) = transMatrix[2][0];
    *out.ptr(index, 7) = transMatrix[2][1];
    *out.ptr(index, 8) = transMatrix[2][2];
}

template<class Transform, class Filter, typename T>
__global__ void warp(const Filter src, Ptr2dVarShapeNHWC<T> dst, const cuda::Tensor2DWrap<float> d_coeffs_)
{
    const int x         = blockDim.x * blockIdx.x + threadIdx.x;
    const int y         = blockDim.y * blockIdx.y + threadIdx.y;
    const int batch_idx = get_batch_idx();
    const int lid       = get_lid();

    extern __shared__ float coeff[];
    if (lid < 9)
    {
        coeff[lid] = *d_coeffs_.ptr(batch_idx, lid);
    }
    __syncthreads();

    if (x < dst.at_cols(batch_idx) && y < dst.at_rows(batch_idx))
    {
        const float2 coord = Transform::calcCoord(coeff, x, y);
        *dst.ptr(batch_idx, y, x)
            = nv::cv::cuda::SaturateCast<nv::cv::cuda::BaseType<T>>(src(batch_idx, coord.y, coord.x));
    }
}

template<class Transform, template<typename> class Filter, template<typename> class B, typename T>
struct WarpDispatcher
{
    static void call(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, const cuda::Tensor2DWrap<float> d_coeffs,
                     const int max_height, const int max_width, const float4 borderValue, cudaStream_t stream)
    {
        using work_type = nv::cv::cuda::ConvertBaseTypeTo<float, T>;

        dim3 block(BLOCK, BLOCK / 4);
        dim3 grid(divUp(max_width, block.x), divUp(max_height, block.y), dst.batches);

        work_type    borderVal = nv::cv::cuda::DropCast<NumComponents<T>>(borderValue);
        B<work_type> brd(0, 0, borderVal);
        // B<work_type> brd(max_height, max_width, borderVal);
        BorderReader<Ptr2dVarShapeNHWC<T>, B<work_type>>         brdSrc(src, brd);
        Filter<BorderReader<Ptr2dVarShapeNHWC<T>, B<work_type>>> filter_src(brdSrc);
        size_t                                                   smem_size = 9 * sizeof(float);
        warp<Transform><<<grid, block, smem_size, stream>>>(filter_src, dst, d_coeffs);
        checkKernelErrors();
    }
};

template<class Transform, typename T>
void warp_caller(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst, const cuda::Tensor2DWrap<float> transform,
                 const int max_height, const int max_width, const int interpolation, const int borderMode,
                 const float4 borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const Ptr2dVarShapeNHWC<T> src, Ptr2dVarShapeNHWC<T> dst,
                           const cuda::Tensor2DWrap<float> transform, const int max_height, const int max_width,
                           const float4 borderValue, cudaStream_t stream);

    static const func_t funcs[3][5] = {
        {WarpDispatcher<Transform,  PointFilter, BrdConstant, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdReplicate, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdReflect, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdWrap, T>::call,
         WarpDispatcher<Transform,  PointFilter, BrdReflect101, T>::call},
        {WarpDispatcher<Transform, LinearFilter, BrdConstant, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdReplicate, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdReflect, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdWrap, T>::call,
         WarpDispatcher<Transform, LinearFilter, BrdReflect101, T>::call},
        {WarpDispatcher<Transform,  CubicFilter, BrdConstant, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdReplicate, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdReflect, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdWrap, T>::call,
         WarpDispatcher<Transform,  CubicFilter, BrdReflect101, T>::call}
    };

    funcs[interpolation][borderMode](src, dst, transform, max_height, max_width, borderValue, stream);
}

template<typename T>
void warpAffine(const void **input, void **output, float *d_coeffs, const int *height, const int *width,
                const int *out_height, const int *out_width, const int max_height, const int max_width, const int batch,
                const int interpolation, int borderMode, const float *borderValue, cudaStream_t stream)
{
    int channels = VecTraits<T>::cn;
    cuda_op::Ptr2dVarShapeNHWC<T> src_ptr(batch, height, width, channels, (T **) input);
    cuda_op::Ptr2dVarShapeNHWC<T> dst_ptr(batch, out_height, out_width, channels, (T **) output);
    warp_caller<AffineTransform, T>(src_ptr, dst_ptr, d_coeffs, max_height, max_width,
                                    interpolation, borderMode, borderValue, stream);
}

template<typename T>
void warpPerspective(const nv::cv::IImageBatchVarShapeDataPitchDevice &inData,
                     const nv::cv::IImageBatchVarShapeDataPitchDevice &outData,
                     const cuda::Tensor2DWrap<float> transform, const int interpolation, const int borderMode,
                     const float4 borderValue, cudaStream_t stream)
{
    Ptr2dVarShapeNHWC<T> src_ptr(inData);
    Ptr2dVarShapeNHWC<T> dst_ptr(outData);

    Size2D outMaxSize = outData.maxSize();

    warp_caller<PerspectiveTransform, T>(src_ptr, dst_ptr, transform, outMaxSize.h, outMaxSize.w, interpolation,
                                         borderMode, borderValue, stream);
}

void invertMatVarShape(cv::Mat M, float *h_aCoeffs)
{
    float den = M.at<float>(0, 0) * M.at<float>(1, 1) - M.at<float>(0, 1) * M.at<float>(1, 0);
    den = std::abs(den) > 1e-5 ? 1. / den : .0;
    h_aCoeffs[0] = (float) M.at<float>(1, 1) * den;
    h_aCoeffs[1] = (float) - M.at<float>(0, 1) * den;
    h_aCoeffs[2] = (float)(M.at<float>(0, 1) * M.at<float>(1, 2) - M.at<float>(1, 1) * M.at<float>(0, 2)) * den;
    h_aCoeffs[3] = (float) - M.at<float>(1, 0) * den;
    h_aCoeffs[4] = (float) M.at<float>(0, 0) * den;
    h_aCoeffs[5] = (float)(M.at<float>(1, 0) * M.at<float>(0, 2) - M.at<float>(0, 0) * M.at<float>(1, 2)) * den;
}

size_t WarpAffineVarShape::calBufferSize(int batch_size)
{
    return (2 * sizeof(void *) + 4 * sizeof(int) + 9 * sizeof(float)) * batch_size;
}

int WarpAffineVarShape::infer(const void **data_in, void **data_out, void *gpu_workspace, void *cpu_workspace,
                              const int batch, const size_t buffer_size, const cv::Size *dsize, const float *trans_matrix,
                              const int flags, const int borderMode, const cv::Scalar borderValue, const DataShape *input_shape,
                              DataFormat format, DataType data_type, cudaStream_t stream)
{
    if(!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = input_shape[0].C;
    const int interpolation = flags & cv::INTER_MAX;

    if(channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    if(!(data_type == kCV_8U || data_type == kCV_8S ||
            data_type == kCV_16U || data_type == kCV_16S ||
            data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    CV_Assert(interpolation == cv::INTER_NEAREST || interpolation == cv::INTER_LINEAR || interpolation == cv::INTER_CUBIC);
    CV_Assert(borderMode == cv::BORDER_REFLECT101 || borderMode == cv::BORDER_REPLICATE
              || borderMode == cv::BORDER_CONSTANT || borderMode == cv::BORDER_REFLECT || borderMode == cv::BORDER_WRAP);

    const void **inputs = (const void **)cpu_workspace;
    void **outputs = (void **)((char *)inputs + sizeof(void *) * batch);
    int *rows = (int *)((char *)outputs + sizeof(void *) * batch);
    int *cols = (int *)((char *)rows + sizeof(int) * batch);
    int *out_rows = (int *)((char *)cols + sizeof(int) * batch);
    int *out_cols = (int *)((char *)out_rows + sizeof(int) * batch);
    float *h_aCoeffs = (float *)((char *)out_cols + sizeof(int) * batch);

    size_t data_size = DataSize(data_type);
    int max_out_width = 0, max_out_height = 0;

    for(int i = 0; i < batch; ++i)
    {
        inputs[i] = data_in[i];
        outputs[i] = data_out[i];
        rows[i] = input_shape[i].H;
        cols[i] = input_shape[i].W;
        out_rows[i] = dsize[i].height;
        out_cols[i] = dsize[i].width;

        cv::Mat coeffsMat(2, 3, CV_32FC1, (void *)(h_aCoeffs + i * 9));
        cv::Mat trans_mat(2, 3, CV_32FC1, (void *)(trans_matrix + i * 6));
        if(flags & cv::WARP_INVERSE_MAP)
        {
            trans_mat.convertTo(coeffsMat, coeffsMat.type());
        }
        else
        {
            invertMatVarShape(trans_mat, h_aCoeffs + i * 9);
        }

        if(max_out_width < dsize[i].width)
            max_out_width = dsize[i].width;
        if(max_out_height < dsize[i].height)
            max_out_height = dsize[i].height;
    }

    const void **inputs_gpu = (const void **)gpu_workspace;
    void **outputs_gpu = (void **)((char *)inputs_gpu + sizeof(void *) * batch);
    int *rows_gpu = (int *)((char *)outputs_gpu + sizeof(void *) * batch);
    int *cols_gpu = (int *)((char *)rows_gpu + sizeof(int) * batch);
    int *out_rows_gpu = (int *)((char *)cols_gpu + sizeof(int) * batch);
    int *out_cols_gpu = (int *)((char *)out_rows_gpu + sizeof(int) * batch);
    float *d_aCoeffs_gpu = (float *)((char *)out_cols_gpu + sizeof(int) * batch);

    checkCudaErrors(cudaMemcpyAsync((void *)gpu_workspace, (void *)cpu_workspace, buffer_size, cudaMemcpyHostToDevice,
                                    stream));

    typedef void (*func_t)(const void **input, void **output, float *d_coeffs, const int *height, const int *width,
                           const int *out_height, const int *out_width, const int max_height, const int max_width, const int batch,
                           const int interpolation, int borderMode, const float *borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {warpAffine<uchar>, 0 /*warpAffine<uchar2>*/, warpAffine<uchar3>, warpAffine<uchar4>     },
        {0 /*warpAffine<schar>*/, 0 /*warpAffine<char2>*/, 0 /*warpAffine<char3>*/, 0 /*warpAffine<char4>*/},
        {warpAffine<ushort>, 0 /*warpAffine<ushort2>*/, warpAffine<ushort3>, warpAffine<ushort4>    },
        {warpAffine<short>, 0 /*warpAffine<short2>*/, warpAffine<short3>, warpAffine<short4>     },
        {0 /*warpAffine<int>*/, 0 /*warpAffine<int2>*/, 0 /*warpAffine<int3>*/, 0 /*warpAffine<int4>*/ },
        {warpAffine<float>, 0 /*warpAffine<float2>*/, warpAffine<float3>, warpAffine<float4>     }
    };

    const func_t func = funcs[data_type][channels - 1];
    CV_Assert(func != 0);

    cv::Scalar_<float> borderValueFloat;
    borderValueFloat = borderValue;

    func(inputs_gpu, outputs_gpu, d_aCoeffs_gpu, rows_gpu, cols_gpu, out_rows_gpu, out_cols_gpu, max_out_height,
         max_out_width,
         batch, interpolation, borderMode, borderValueFloat.val, stream);
    return 0;
}

WarpPerspectiveVarShape::WarpPerspectiveVarShape(const int32_t maxBatchSize)
    : CudaBaseOp()
    , m_maxBatchSize(maxBatchSize)
{
    if (m_maxBatchSize > 0)
    {
        size_t bufferSize = sizeof(float) * 9 * m_maxBatchSize;
        NVCV_CHECK_LOG(cudaMalloc(&m_transformationMatrix, bufferSize));
    }
}

WarpPerspectiveVarShape::~WarpPerspectiveVarShape()
{
    if (m_transformationMatrix != nullptr)
    {
        NVCV_CHECK_LOG(cudaFree(m_transformationMatrix));
    }
    m_transformationMatrix = nullptr;
}

ErrorCode WarpPerspectiveVarShape::infer(const IImageBatchVarShapeDataPitchDevice &inData,
                                         const IImageBatchVarShapeDataPitchDevice &outData,
                                         const ITensorDataPitchDevice &transMatrix, const int32_t flags,
                                         const NVCVBorderType borderMode, const float4 borderValue, cudaStream_t stream)
{
    if (m_maxBatchSize <= 0)
    {
        LOG_ERROR("Operator warp perspective var shape is not initialized properly, maxVarShapeBatchSize: "
                  << m_maxBatchSize);
        return ErrorCode::INVALID_PARAMETER;
    }

    if (m_maxBatchSize < inData.numImages())
    {
        LOG_ERROR("Invalid number of images, it should not exceed " << m_maxBatchSize);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

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

    if (!inData.uniqueFormat())
    {
        LOG_ERROR("Images in the input varshape must all have the same format");
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = inData.uniqueFormat().numChannels();

    if (channels > 4)
    {
        LOG_ERROR("Invalid channel number " << channels);
        return ErrorCode::INVALID_DATA_SHAPE;
    }

    const int interpolation = flags & NVCV_INTERP_MAX;

    DataType data_type = helpers::GetLegacyDataType(inData.uniqueFormat());

    if (!(data_type == kCV_8U || data_type == kCV_8S || data_type == kCV_16U || data_type == kCV_16S
          || data_type == kCV_32S || data_type == kCV_32F))
    {
        LOG_ERROR("Invalid DataType " << data_type);
        return ErrorCode::INVALID_DATA_TYPE;
    }

    NVCV_ASSERT(interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
                || interpolation == NVCV_INTERP_CUBIC);
    NVCV_ASSERT(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
                || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT
                || borderMode == NVCV_BORDER_WRAP);

    // Check if inverse op is needed
    bool performInverse = flags & NVCV_WARP_INVERSE_MAP;

    // Wrap the matrix in 2D wrappers with proper pitch
    cuda::Tensor2DWrap<float> transMatrixInput(transMatrix);
    cuda::Tensor2DWrap<float> transMatrixOutput(m_transformationMatrix, static_cast<int>(sizeof(float) * 9));

    if (performInverse)
    {
        inverseMatWarpPerspective<<<1, inData.numImages(), 0, stream>>>(inData.numImages(), transMatrixInput,
                                                                        transMatrixOutput);
        checkKernelErrors();
    }
    else
    {
        NVCV_CHECK_LOG(cudaMemcpy2DAsync(m_transformationMatrix, sizeof(float) * 9, transMatrixInput.ptr(0, 0),
                                         transMatrixInput.pitchBytes()[0], sizeof(float) * 9, inData.numImages(),
                                         cudaMemcpyDeviceToDevice, stream));
    }

    typedef void (*func_t)(const nv::cv::IImageBatchVarShapeDataPitchDevice &inData,
                           const nv::cv::IImageBatchVarShapeDataPitchDevice &outData,
                           const cuda::Tensor2DWrap<float> transform, const int interpolation, const int borderMode,
                           const float4 borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        {      warpPerspective<uchar>,  0 /*warpPerspective<uchar2>*/,      warpPerspective<uchar3>,warpPerspective<uchar4>                                                                                                    },
        {0 /*warpPerspective<schar>*/,   0 /*warpPerspective<char2>*/, 0 /*warpPerspective<char3>*/,
         0 /*warpPerspective<char4>*/                                                                                        },
        {     warpPerspective<ushort>, 0 /*warpPerspective<ushort2>*/,     warpPerspective<ushort3>, warpPerspective<ushort4>},
        {      warpPerspective<short>,  0 /*warpPerspective<short2>*/,      warpPerspective<short3>,  warpPerspective<short4>},
        {  0 /*warpPerspective<int>*/,    0 /*warpPerspective<int2>*/,  0 /*warpPerspective<int3>*/,
         0 /*warpPerspective<int4>*/                                                                                         },
        {      warpPerspective<float>,  0 /*warpPerspective<float2>*/,      warpPerspective<float3>,  warpPerspective<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    func(inData, outData, transMatrixOutput, interpolation, borderMode, borderValue, stream);
    return SUCCESS;
}

} // namespace nv::cv::legacy::cuda_op
