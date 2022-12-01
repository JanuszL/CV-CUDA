/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * Copyright (c) 2021-2022, Bytedance Inc. All rights reserved.
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

namespace nvcv = nv::cv;

template<class Transform, class Filter, typename T>
__global__ void warp(const Filter src, Ptr2dNHWC<T> dst, Transform transform)
{
    const int               x         = blockDim.x * blockIdx.x + threadIdx.x;
    const int               y         = blockDim.y * blockIdx.y + threadIdx.y;
    const int               lid       = get_lid();
    const int               batch_idx = get_batch_idx();
    extern __shared__ float coeff[];
    if (lid < 9)
    {
        coeff[lid] = transform.xform[lid];
    }
    __syncthreads();
    if (x < dst.cols && y < dst.rows)
    {
        const float2 coord        = Transform::calcCoord(coeff, x, y);
        *dst.ptr(batch_idx, y, x) = nvcv::cuda::SaturateCast<nvcv::cuda::BaseType<T>>(src(batch_idx, coord.y, coord.x));
    }
}

template<class Transform, template<typename> class Filter, template<typename> class B, typename T>
struct WarpDispatcher
{
    static void call(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, Transform transform, const float4 borderValue,
                     cudaStream_t stream)
    {
        using work_type = nv::cv::cuda::ConvertBaseTypeTo<float, T>;

        dim3 block(BLOCK, BLOCK / 4);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y), dst.batches);

        work_type                                borderVal = nv::cv::cuda::DropCast<NumComponents<T>>(borderValue);
        B<work_type>                             brd(src.rows, src.cols, borderVal);
        BorderReader<Ptr2dNHWC<T>, B<work_type>> brdSrc(src, brd);
        Filter<BorderReader<Ptr2dNHWC<T>, B<work_type>>> filter_src(brdSrc);
        size_t                                           smem_size = 9 * sizeof(float);
        warp<Transform><<<grid, block, smem_size, stream>>>(filter_src, dst, transform);
        checkKernelErrors();
    }
};

template<class Transform, typename T>
void warp_caller(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, Transform transform, int interpolation, int borderMode,
                 const float4 borderValue, cudaStream_t stream)
{
    typedef void (*func_t)(const Ptr2dNHWC<T> src, Ptr2dNHWC<T> dst, Transform transform, const float4 borderValue,
                           cudaStream_t stream);

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

    funcs[interpolation][borderMode](src, dst, transform, borderValue, stream);
}

template<typename T>
void warpAffine(const nvcv::TensorDataAccessPitchImagePlanar &inData,
                const nvcv::TensorDataAccessPitchImagePlanar &outData, WarpAffineTransform transform,
                const int interpolation, int borderMode, const float4 borderValue, cudaStream_t stream)
{
    Ptr2dNHWC<T> src_ptr(inData);
    Ptr2dNHWC<T> dst_ptr(outData);
    warp_caller<WarpAffineTransform, T>(src_ptr, dst_ptr, transform, interpolation, borderMode, borderValue, stream);
}

template<typename T>
void warpPerspective(const void *input, void *output, float *d_coeffs, cuda_op::DataShape input_shape,
                     const float *trans_matrix, const cv::Size dsize, const int interpolation, int borderMode,
                     const float *borderValue, cudaStream_t stream)
{
    cuda_op::Ptr2dNHWC<T> src_ptr(input_shape.N, input_shape.H, input_shape.W, input_shape.C, (T *) input);
    cuda_op::Ptr2dNHWC<T> dst_ptr(input_shape.N, dsize.height, dsize.width, input_shape.C, (T *) output);
    checkCudaErrors(cudaMemcpyAsync(d_coeffs, trans_matrix, 9 * sizeof(float), cudaMemcpyHostToDevice, stream));
    warp_caller<PerspectiveTransform, T>(src_ptr, dst_ptr, d_coeffs, interpolation, borderMode, borderValue, stream);
}

static void invertMat(const float *M, float *h_aCoeffs)
{
    // M is stored in row-major format M[0,0], M[0,1], M[0,2], M[1,0], M[1,1], M[1,2]
    float den    = M[0] * M[4] - M[1] * M[3];
    den          = std::abs(den) > 1e-5 ? 1. / den : .0;
    h_aCoeffs[0] = (float)M[5] * den;
    h_aCoeffs[1] = (float)-M[1] * den;
    h_aCoeffs[2] = (float)(M[1] * M[5] - M[4] * M[2]) * den;
    h_aCoeffs[3] = (float)-M[3] * den;
    h_aCoeffs[4] = (float)M[0] * den;
    h_aCoeffs[5] = (float)(M[3] * M[2] - M[0] * M[5]) * den;
}

namespace nv::cv::legacy::cuda_op {

ErrorCode WarpAffine::infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                            const float *xform, const int flags, const NVCVBorderType borderMode,
                            const float4 borderValue, cudaStream_t stream)
{
    DataFormat input_format  = GetLegacyDataFormat(inData.layout());
    DataFormat output_format = GetLegacyDataFormat(outData.layout());

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

    auto inAccess = TensorDataAccessPitchImagePlanar::Create(inData);
    NVCV_ASSERT(inAccess);

    auto outAccess = TensorDataAccessPitchImagePlanar::Create(outData);
    NVCV_ASSERT(outAccess);

    cuda_op::DataType  data_type   = GetLegacyDataType(inData.dtype());
    cuda_op::DataShape input_shape = GetLegacyDataShape(inAccess->infoShape());

    int       channels      = input_shape.C;
    const int interpolation = flags & NVCV_INTERP_MAX;

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

    NVCV_ASSERT(interpolation == NVCV_INTERP_NEAREST || interpolation == NVCV_INTERP_LINEAR
                || interpolation == NVCV_INTERP_CUBIC);
    NVCV_ASSERT(borderMode == NVCV_BORDER_REFLECT101 || borderMode == NVCV_BORDER_REPLICATE
                || borderMode == NVCV_BORDER_CONSTANT || borderMode == NVCV_BORDER_REFLECT
                || borderMode == NVCV_BORDER_WRAP);

    typedef void (*func_t)(const nvcv::TensorDataAccessPitchImagePlanar &inData,
                           const nvcv::TensorDataAccessPitchImagePlanar &outData, WarpAffineTransform transform,
                           const int interpolation, int borderMode, const float4 borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] = {
        { warpAffine<uchar>, 0,  warpAffine<uchar3>,  warpAffine<uchar4>},
        {                 0, 0,                   0,                   0},
        {warpAffine<ushort>, 0, warpAffine<ushort3>, warpAffine<ushort4>},
        { warpAffine<short>, 0,  warpAffine<short3>,  warpAffine<short4>},
        {                 0, 0,                   0,                   0},
        { warpAffine<float>, 0,  warpAffine<float3>,  warpAffine<float4>}
    };

    const func_t func = funcs[data_type][channels - 1];
    NVCV_ASSERT(func != 0);

    WarpAffineTransform transform;

    // initialize affine transform
    for (int i = 0; i < 9; i++)
    {
        transform.xform[i] = i < 6 ? (float)(xform[i]) : 0.0f;
    }

    if (flags & NVCV_WARP_INVERSE_MAP)
    {
        invertMat(xform, transform.xform);
    }

    func(*inAccess, *outAccess, transform, interpolation, borderMode, borderValue, stream);

    return ErrorCode::SUCCESS;
}

size_t WarpPerspective::calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
{
    return 9 * sizeof(float);
}

int WarpPerspective::infer(const void *const *inputs, void **outputs, void *workspace, const float trans_matrix[3 * 3],
                           void *cpu_workspace, const cv::Size dsize, const int flags, const int borderMode, const cv::Scalar borderValue,
                           DataShape input_shape, DataFormat format, DataType data_type, cudaStream_t stream)
{
    if(!(format == kNHWC || format == kHWC))
    {
        LOG_ERROR("Invalid DataFormat " << format);
        return ErrorCode::INVALID_DATA_FORMAT;
    }

    int channels = input_shape.C;
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

    typedef void (*func_t)(const void *input, void *output, float* d_coeffs, cuda_op::DataShape input_shape,
                           const float *trans_matrix, const cv::Size dsize, const int interpolation, int borderMode,
                           const float* borderValue, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {warpPerspective<uchar>, 0 /*warpPerspective<uchar2>*/, warpPerspective<uchar3>, warpPerspective<uchar4>     },
        {0 /*warpPerspective<schar>*/, 0 /*warpPerspective<char2>*/, 0 /*warpPerspective<char3>*/, 0 /*warpPerspective<char4>*/},
        {warpPerspective<ushort>, 0 /*warpPerspective<ushort2>*/, warpPerspective<ushort3>, warpPerspective<ushort4>    },
        {warpPerspective<short>, 0 /*warpPerspective<short2>*/, warpPerspective<short3>, warpPerspective<short4>     },
        {0 /*warpPerspective<int>*/, 0 /*warpPerspective<int2>*/, 0 /*warpPerspective<int3>*/, 0 /*warpPerspective<int4>*/ },
        {warpPerspective<float>, 0 /*warpPerspective<float2>*/, warpPerspective<float3>, warpPerspective<float4>     }
    };

    const func_t func = funcs[data_type][channels - 1];
    CV_Assert(func != 0);

    cv::Mat coeffsMat(3, 3, CV_32FC1, (void *)cpu_workspace);
    cv::Mat trans_mat(3, 3, CV_32FC1, (void *)trans_matrix);
    if(flags & cv::WARP_INVERSE_MAP)
    {
        trans_mat.convertTo(coeffsMat, coeffsMat.type());
    }
    else
    {
        cv::Mat iM;
        cv::invert(trans_mat, iM);
        iM.convertTo(coeffsMat, coeffsMat.type());
    }

    cv::Scalar_<float> borderValueFloat;
    borderValueFloat = borderValue;

    func(inputs[0], outputs[0], (float *)workspace, input_shape, (float *)cpu_workspace, dsize, interpolation, borderMode,
         borderValueFloat.val, stream);
    return 0;
}

} // namespace nv::cv::legacy::cuda_op
