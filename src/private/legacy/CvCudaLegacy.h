/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
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

#ifndef CV_CUDA_LEGACY_H
#define CV_CUDA_LEGACY_H

#include <cuda_runtime.h>
#include <nvcv/IImageBatch.hpp>
#include <nvcv/IImageBatchData.hpp>
#include <nvcv/ITensorData.hpp>
#include <nvcv/Rect.h>
#include <operators/Types.h>
#include <util/Assert.h>

#include <vector>

namespace nv::cv::legacy::cuda_op {

enum ErrorCode
{
    SUCCESS             = 0,
    INVALID_DATA_TYPE   = 1,
    INVALID_DATA_SHAPE  = 2,
    INVALID_DATA_FORMAT = 3,
    INVALID_PARAMETER   = 4
};

enum DataFormat
{
    kNCHW = 0,
    kNHWC = 1,
    kCHW  = 2,
    kHWC  = 3,
};

enum DataType
{
    kCV_8U  = 0,
    kCV_8S  = 1,
    kCV_16U = 2,
    kCV_16S = 3,
    kCV_32S = 4,
    kCV_32F = 5,
    kCV_64F = 6,
    kCV_16F = 7,
};

struct DataShape
{
    DataShape()
        : N(1)
        , C(0)
        , H(0)
        , W(0){};
    DataShape(int n, int c, int h, int w)
        : N(n)
        , C(c)
        , H(h)
        , W(w){};
    DataShape(int c, int h, int w)
        : N(1)
        , C(c)
        , H(h)
        , W(w){};
    int N = 1; // batch
    int C;     // channel
    int H;     // height
    int W;     // width
};

inline size_t DataSize(DataType data_type)
{
    size_t size = 0;
    switch (data_type)
    {
    case kCV_8U:
    case kCV_8S:
        size = 1;
        break;
    case kCV_16U:
    case kCV_16S:
    case kCV_16F:
        size = 2;
        break;
    case kCV_32S:
    case kCV_32F:
        size = 4;
        break;
    case kCV_64F:
        size = 8;
        break;
    default:
        break;
    }
    return size;
}

// cuda base operator class
class CudaBaseOp
{
public:
    CudaBaseOp(){};

    CudaBaseOp(DataShape max_input_shape, DataShape max_output_shape)
        : max_input_shape_(max_input_shape)
        , max_output_shape_(max_output_shape)
    {
    }

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type)
    {
        return 0;
    };

    bool checkDataShapeValid(DataShape input_shape, DataShape output_shape)
    {
        int input_size      = input_shape.N * input_shape.C * input_shape.H * input_shape.W;
        int max_input_size  = max_input_shape_.N * max_input_shape_.C * max_input_shape_.H * max_input_shape_.W;
        int output_size     = output_shape.N * output_shape.C * output_shape.H * output_shape.W;
        int max_output_size = max_output_shape_.N * max_output_shape_.C * max_output_shape_.H * max_output_shape_.W;
        return (input_size <= max_input_size) && (output_size <= max_output_size);
    }

protected:
    DataShape max_input_shape_;
    DataShape max_output_shape_;
};

class ConvertTo : public CudaBaseOp
{
public:
    ConvertTo() = delete;

    ConvertTo(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Converts an array to another data type with scaling.
     * The method converts source pixel values to the target data type. saturate_cast<> is applied at the end to avoid
     * possible overflows:
     *
     * ```
     * outputs(x,y) = saturate_cast<out_type>(α * inputs(x, y) + β)
     * ```
     *
     * Limitations:
     *
     * Data Layout, Number, Channels, Width, Height, of input and output must be same.
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1-4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1-4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | No
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     *
     *
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the same shape as input_shape and the
     * type out_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param out_type desired output type.
     * @param alpha scale factor.
     * @param beta shift data added to the scaled values.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     *
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, const double alpha,
                    const double beta, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
};

class CustomCrop : public CudaBaseOp
{
public:
    CustomCrop() = delete;

    CustomCrop(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Crops the a given input image into a destination image.
     *        Destination will have the [0,0] position populated by the x,y position as
     *        defined in the ROI x,y parameters of the input data. The operator will continue to populate the
     *        output data until the destination image is populated with the size described by the ROI.
     *
     *
     * Limitations:
     *
     * ROI must be smaller than output tensor.
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | Yes
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | Yes
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     *
     * @param [in] in intput tensor.
     *
     * @param [out] out output tensor.
     * @param [in]  roi region of interest, defined in pixels
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, NVCVRectI roi,
                    cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
};

class Reformat : public CudaBaseOp
{
public:
    Reformat() = delete;

    Reformat(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Reformats the input images. Transfor the inputs from kNHWC format to kNCHW format or from kNCHW format to
     * kNHWC format.
     *
     * Limitations:
     *
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC, kNCHW, KCHW]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC, kNCHW, KCHW]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | Yes
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | No
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the same shape as input_shape and the
     * same type as data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param input_shape shape of the input images.
     * @param input_format input format. kNHWC -> kNCHW, kNCHW -> kNHWC.
     * @param output_format output format.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
    void      checkDataFormat(DataFormat format);
};

class Resize : public CudaBaseOp
{
public:
    Resize() = delete;

    Resize(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Resizes the input images. This class resizes the images down to or up to the specified size.
     *
     *
     * Limitations:
     *
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     * @param [in] inData Intput tensor.
     * @param [out] outData Output tensor.
     * @param [in] interpolation Interpolation method. See \ref NVCVInterpolationType for more details.
     * @param [in] stream Stream for the asynchronous execution.
     *
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                    const NVCVInterpolationType interpolation, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
};

class Morphology : public CudaBaseOp
{
public:
    Morphology() = delete;

    Morphology(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Dilates/Erodes an image
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | No
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     *
     * @param inData gpuData to a tensor of one or more HWC images
     * @param outData gpuData a tensor hosting the outputs of the operation
     * @param morph_type Type of operation to perform on data Erode/Dilate
     * @param mask_size shape and size of the mask to use for the operation
     * @param anchor anchor to use for the kernel (-1,-1) will use center of kernel
     * @param iteraton number of times to perform the kernel pass
     * @param borderMode the border mode to use when acessing data outside of source
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                    NVCVMorphologyType morph_type, Size2D mask_size, int2 anchor, int iteration,
                    const NVCVBorderType borderMode, cudaStream_t stream);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     * @param max_mask_size maximum mask size that may be used
     * @param max_iteration maximum iteration that may be used
     */
    size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type,
                         Size2D max_mask_size, int max_iteration);
};

class Normalize : public CudaBaseOp
{
public:
    Normalize() = delete;

    Normalize(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Data normalization is done using externally provided base (typically: mean or min) and scale (typically
     * reciprocal of standard deviation or 1/(max-min)). The normalization follows the formula:
     * ```
     * out[data_idx] = (in[data_idx] - base[param_idx]) * scale[param_idx] * global_scale + shift
     * ```
     * Where `data_idx` is a position in the data tensor (in, out) and `param_idx` is a position
     * in the base and scale tensors (see below for details). The two additional constants,
     * `global_scale` and `shift` can be used to adjust the result to the dynamic range and resolution
     * of the output type.
     *
     * The `scale` parameter may also be interpreted as standard deviation - in that case, its
     * reciprocal is used and optionally, a regularizing term is added to the variance.
     * ```
     * m = 1 / sqrt(square(stddev[param_idx]) + epsilon)
     * out[data_idx] = (in[data_idx] - mean[param_idx]) * m * global_scale + shift
     * ```
     *
     * `param_idx` is calculated as follows (where axis = N,H,W,C):
     * ```
     * param_idx[axis] = param_shape[axis] == 1 ? 0 : data_idx[axis]
     * ```
     *
     * Limitations:
     *
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC, kNCHW, KCHW]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC, kNCHW, KCHW]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | Yes
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * Scale/Base Tensor:
     *
     * Scale and Base may be a tensor the same shape as the input/output tensors, or it can be a scalar each dimension.
     *
     *
     * @param inputs gpu pointer,
     * @param global_scale additional scaling factor, used e.g. when output is of integral type.
     * @param shift additional bias value, used e.g. when output is of unsigned type.
     * @param epsilon regularizing term added to variance; only used if scale_is_stddev = true
     * @param flags if true, scale is interpreted as standard deviation and it's regularized and its
     * reciprocal is used when scaling.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &baseData,
                    const ITensorDataPitchDevice &scaleData, const ITensorDataPitchDevice &outData,
                    const float global_scale, const float shift, const float epsilon, const uint32_t flags,
                    cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
    void      checkParamShape(DataShape input_shape, DataShape param_shape);
};

class PadAndStack : public CudaBaseOp
{
public:
    PadAndStack() = delete;

    PadAndStack(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     * Top/left Tensors
     *
     *     Must be kNHWC where N=H=C=1 with W = N (N in reference to input and output tensors).
     *     Data Type must be 32bit Signed.
     */

    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                    const ITensorDataPitchDevice &top, const ITensorDataPitchDevice &left,
                    const NVCVBorderType borderMode, const float borderValue, cudaStream_t stream);

    size_t calBufferSize(int batch_size);
};

class Rotate : public CudaBaseOp
{
public:
    Rotate() = delete;
    Rotate(DataShape max_input_shape, DataShape max_output_shape);

    ~Rotate();

    /**
     * @brief Rotates input images around the origin (0,0) and then shifts it.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param dsize size of the output images.
     * @param angle angle of rotation in degrees.
     * @param xShift shift along the horizontal axis.
     * @param yShift shift along the vertical axis.
     * @param interpolation interpolation method. Only INTER_NEAREST, INTER_LINEAR, and INTER_CUBIC are supported.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, const double angleDeg,
                    const double2 shift, const NVCVInterpolationType interpolation, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);

protected:
    double *d_aCoeffs;
};

class MedianBlur : public CudaBaseOp
{
public:
    MedianBlur() = delete;

    MedianBlur(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Blur an image using a median kernel.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param ksize median blur kernel size.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, const cv::Size2D ksize,
                    cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
};

class NormalizeVarShape : public CudaBaseOp
{
public:
    NormalizeVarShape() = delete;

    NormalizeVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Data normalization is done using externally provided base (typically: mean or min) and scale (typically
     * reciprocal of standard deviation or 1/(max-min)). The normalization follows the formula:
     * ```
     * out[data_idx] = (in[data_idx] - base[param_idx]) * scale[param_idx] * global_scale + shift
     * ```
     * Where `data_idx` is a position in the data tensor (in, out) and `param_idx` is a position
     * in the base and scale tensors (see below for details). The two additional constants,
     * `global_scale` and `shift` can be used to adjust the result to the dynamic range and resolution
     * of the output type.
     *
     * The `scale` parameter may also be interpreted as standard deviation - in that case, its
     * reciprocal is used and optionally, a regularizing term is added to the variance.
     * ```
     * m = 1 / sqrt(square(stddev[param_idx]) + epsilon)
     * out[data_idx] = (in[data_idx] - mean[param_idx]) * m * global_scale + shift
     * ```
     *
     * `param_idx` is calculated as follows:
     * ```
     * param_idx[axis] = param_shape[axis] == 1 ? 0 : data_idx[axis]
     * ```
     *
     * @param inputs gpu pointer, inputs[0] to inputs[batch-1] are input images of different shape, whose shapes are
     * input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the same shape as input_shape and the
     * type out_data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variables.
     * @param batch batch_size
     * @param buffer_size buffer size of gpu_workspace and cpu_workspace
     * @param base value(s) to be subtracted from input elements.
     * @param scale value(s) of scales (or standard deviations).
     * @param input_shape shape of the input images.
     * @param base_channel channels of base. base can be a scalar or a batch of tensors with the same dimensionality as
     * the input. The extent in each dimension must match the value of the input or be equal to 1. If the extent is 1,
     * the value will be broadcast in this dimension.
     * @param scale_channel channels of scale. See base_param_shape argument for more information about shape
     * constraints.
     * @param scale_is_stddev if true, scale is interpreted as standard deviation and it's regularized and its
     * reciprocal is used when scaling.
     * @param global_scale additional scaling factor, used e.g. when output is of integral type.
     * @param shift additional bias value, used e.g. when output is of unsigned type.
     * @param epsilon regularizing term added to variance; only used if scale_is_stddev = true
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param out_data_type data type of the output images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const nv::cv::IImageBatchVarShapeDataPitchDevice &inData,
                    const nv::cv::ITensorDataPitchDevice &baseData, const nv::cv::ITensorDataPitchDevice &scaleData,
                    const nv::cv::IImageBatchVarShapeDataPitchDevice &outData, const float global_scale,
                    const float shift, const float epsilon, const uint32_t flags, cudaStream_t stream);
};

class ResizeVarShape : public CudaBaseOp
{
public:
    ResizeVarShape() = delete;

    ResizeVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Resizes the input images. The function resize resizes the image down to or up to the specified size.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the same type as data_type. The output
     * sizes are derived from the dsize,fx, and fy.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variables.
     * @param batch batch_size.
     * @param buffer_size buffer size of gpu_workspace and cpu_workspace
     * @param dsize size of the output images.if it equals zero, it is computed as:
     * ```
     * dsize = Size(round(fx*src.cols), round(fy*src.rows)). Either dsize or both fx and fy must be non-zero.
     * ```
     * @param fx scale factor along the horizontal axis; when it equals 0, it is computed as
     * ```
     * (double)dsize.width/src.cols
     * ```
     * @param fy scale factor along the vertical axis; when it equals 0, it is computed as
     * ```
     * (double)dsize.height/src.rows
     * ```
     * @param interpolation interpolation method. See cv::InterpolationFlags for more detials.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     *
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const NVCVInterpolationType interpolation, cudaStream_t stream);
};

class CopyMakeBorder : public CudaBaseOp
{
public:
    CopyMakeBorder() = delete;

    CopyMakeBorder(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | No
     *      Height        | No
     *
     * @brief Forms a border around an image.
     * The function copies the source image into the middle of the destination image. The areas to the left, to the
     * right, above and below the copied source image will be filled with extrapolated pixels. This is not what
     * filtering functions based on it do (they extrapolate pixels on-fly), but what other more complex functions,
     * including your own, may do to simplify image boundary handling.
     * @param inData Input Tensor
     * @param outData Output Tensor
     * @param top the top pixels.
     * @param left the left pixels.
     * Parameter specifying how many pixels in each direction from the source image rectangle to extrapolate.
     * The src and dist size can be got from input and output tensor.
     * For example, top=1, left=1, src_w=64, src_h=64, dist_w=66, dist_h=66 mean that it builds 1 pixel-wide border.
     * @param border_type border type. See \p NVCVBorderType for details.
     * @param value border value if borderType==BORDER_CONSTANT.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, const int top,
                    const int left, const NVCVBorderType border_type, const float4 value, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
};

class CopyMakeBorderVarShape : public CudaBaseOp
{
public:
    CopyMakeBorderVarShape() = delete;

    CopyMakeBorderVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Forms a border around an image.
     * The function copies the source image into the middle of the destination image. The areas to the left, to the
     * right, above and below the copied source image will be filled with extrapolated pixels. This is not what
     * filtering functions based on it do (they extrapolate pixels on-fly), but what other more complex functions,
     * including your own, may do to simplify image boundary handling.
     * @param inData Input Tensor
     * @param outData Output Tensor
     * @param top the top pixels.
     * @param left the left pixels.
     * Parameter specifying how many pixels in each direction from the source image rectangle to extrapolate.
     * The src and dist size can be got from input and output tensor.
     * For example, top=1, left=1, src_w=64, src_h=64, dist_w=66, dist_h=66 mean that it builds 1 pixel-wide border of top=left=bottom=right=1.
     * @param border_type border type. See NVCVBorderType for details.
     * @param value border value if borderType==BORDER_CONSTANT.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const nv::cv::ITensorDataPitchDevice &top, const nv::cv::ITensorDataPitchDevice &left,
                    const NVCVBorderType border_type, const float4 value, cudaStream_t stream);

    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                    const nv::cv::ITensorDataPitchDevice &top, const nv::cv::ITensorDataPitchDevice &left,
                    const NVCVBorderType border_type, const float4 value, cudaStream_t stream);

private:
    template<class OutType>
    ErrorCode inferWarp(const IImageBatchVarShapeDataPitchDevice &inData, const OutType &outData,
                        const nv::cv::ITensorDataPitchDevice &top, const nv::cv::ITensorDataPitchDevice &left,
                        const NVCVBorderType border_type, const float4 value, cudaStream_t stream);
};

class CenterCrop : public CudaBaseOp
{
public:
    CenterCrop() = delete;

    CenterCrop(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * @brief Crops the given image at the center based on input crop dimensions.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param crop_rows desired number of rows of the crop
     * @param crop_columns desired number of columns of the crop
     * @param input_shape shape of the input images.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, int crop_rows,
                    int crop_columns, cudaStream_t stream);
    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t    calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
};

class RotateVarShape : public CudaBaseOp
{
public:
    RotateVarShape() = delete;

    RotateVarShape(const int maxVarShapeBatchSize);

    ~RotateVarShape();

    /**
     * @brief Rotates input images around the origin (0,0) and then shifts it.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variables.
     * @param batch batch_size.
     * @param buffer_size buffer size of gpu_workspace and cpu_workspace
     * @param dsize size of the output images.
     * @param angle angle of rotation in degrees.
     * @param xShift shift along the horizontal axis.
     * @param yShift shift along the vertical axis.
     * @param interpolation interpolation method. Only INTER_NEAREST, INTER_LINEAR, and INTER_CUBIC are supported.
     * @param input_shape shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const ITensorDataPitchDevice &angleDeg, const ITensorDataPitchDevice &shift,
                    const NVCVInterpolationType interpolation, cudaStream_t stream);

protected:
    double   *d_aCoeffs;
    const int m_maxBatchSize;
};

class Laplacian : public CudaBaseOp
{
public:
    Laplacian() = delete;

    Laplacian(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Calculates the Laplacian of an image.
     * @param inData Input Tensor
     * @param outData Output Tensor
     * @param ksize aperture size used to compute the second-derivative filters, it can be 1 or 3.
     * @param scale optional scale factor for the computed Laplacian values. By default, no scaling is applied.
     * @param borderMode pixel extrapolation method, e.g. \p NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, int ksize, float scale,
                    NVCVBorderType borderMode, cudaStream_t stream);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     */
    size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type);
};

class Gaussian : public CudaBaseOp
{
public:
    Gaussian() = delete;

    Gaussian(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize);

    ~Gaussian();

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blurs an image using a Gaussian filter.
     * @param inputs gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outputs gpu pointer, outputs[0] are batched output images that have the size dsize and the same type as
     * data_type.
     * @param workspace gpu pointer, gpu memory used to store the temporary variables.
     * @param sigma Gaussian kernel standard deviation in X and Y directions.
     *              If sigma.y is zero or negative, use sigma.y = sigma.x.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, Size2D kernelSize,
                    double2 sigma, NVCVBorderType borderMode, cudaStream_t stream);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     * @param maxKernelSize Maximum Gaussian kernel size that may be used
     */
    size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type,
                         Size2D maxKernelSize);

private:
    Size2D  m_maxKernelSize = {0, 0};
    Size2D  m_curKernelSize = {0, 0};
    double2 m_curSigma      = {-1.0, -1.0};
    float  *m_kernel        = nullptr;
};

class Erase : public CudaBaseOp
{
public:
    Erase() = delete;

    Erase(DataShape max_input_shape, DataShape max_output_shape, int num_erasing_area);

    ~Erase();

    /**
     * @brief erase areas of images. Different images in the same batch can be erased differently.
     * @param inData gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
     * @param outData gpu pointer, outputs[0] are batched output images that have the same type as data_type.
     * @param anchor an array of size num_erasing_area that gives the x coordinate and y coordinate of the top left point in the eraseing areas.
     * @param erasing_w an array of size num_erasing_area that gives the widths of the eraseing areas.
     * @param erasing_h an array of size num_erasing_area that gives the heights of the eraseing areas.
     * @param erasing_c an array of size num_erasing_area that gives integers in range 0-15,
            each of whose bits indicates whether or not the corresponding channel need to be erased.
     * @param values an array of size num_erasing_area*4 that gives the filling value for each erase area.
     * @param imgIdx an array of size num_erasing_area that maps a erase area idx to img idx in the batch.
     * @param random an boolean for random op.
     * @param seed random seed for random filling erase area
     * @param inplace for perform inplace op.
     * @param stream for the asynchronous execution.
     *
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData,
                    const ITensorDataPitchDevice &anchor, const ITensorDataPitchDevice &erasing,
                    const ITensorDataPitchDevice &values, const ITensorDataPitchDevice &imgIdx, bool random,
                    unsigned int seed, bool inplace, cudaStream_t stream);

protected:
    int3  *d_max_values;
    void  *temp_storage;
    size_t storage_bytes;
    int    max_num_erasing_area;
};

class AverageBlur : public CudaBaseOp
{
public:
    AverageBlur() = delete;

    AverageBlur(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize);

    ~AverageBlur();

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blur an image using an average kernel.
     * @param ksize average blur kernel size.
     * @param anchor anchor of the kernel that indicates the relative position of a filtered point within the kernel.
     * (-1,-1) means that the anchor is at the kernel center.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const ITensorDataPitchDevice &inData, const ITensorDataPitchDevice &outData, Size2D kernelSize,
                    int2 kernelAnchor, NVCVBorderType borderMode, cudaStream_t stream);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param max_input_shape maximum input DataShape that may be used
     * @param max_output_shape maximum output DataShape that may be used
     * @param max_data_type DataType with the maximum size that may be used
     * @param maxKernelSize Maximum average blur kernel size that may be used
     */
    size_t calBufferSize(DataShape max_input_shape, DataShape max_output_shape, DataType max_data_type,
                         Size2D maxKernelSize);

private:
    Size2D m_maxKernelSize = {0, 0};
    Size2D m_curKernelSize = {0, 0};
    float *m_kernel        = nullptr;
};

class Conv2DVarShape : public CudaBaseOp
{
public:
    Conv2DVarShape() = delete;

    Conv2DVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Convolves an image with the kernel. The function does actually compute correlation, not the convolution
     * (same as opencv filter2D)
     * @param inputs gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outputs gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is
     * input_shape[i] and type is data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variable.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variable.
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param ksize convolution kernel size.
     * @param kernels convolution kernels. All the kernel values are flatted into a 1d array.
     * @param anchors anchor of the kernel that indicates the relative position of a filtered point within the kernel.
     * (-1,-1) means that the anchor is at the kernel center.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const IImageBatchVarShapeDataPitchDevice &kernelData,
                    const ITensorDataPitchDevice &kernelAnchorData, NVCVBorderType borderMode, cudaStream_t stream);
};

class LaplacianVarShape : public CudaBaseOp
{
public:
    LaplacianVarShape() = delete;

    LaplacianVarShape(DataShape max_input_shape, DataShape max_output_shape)
        : CudaBaseOp(max_input_shape, max_output_shape)
    {
    }

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | No
     *      32bit Unsigned | No
     *      32bit Signed   | No
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Calculates the Laplacian of an image.
     * @param inData Input image batch var shape
     * @param outData Output image batch var shape
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param ksize aperture size used to compute the second-derivative filters
     * @param scale optional scale factor for the computed Laplacian values. By default, no scaling is applied.
     * @param borderMode pixel extrapolation method, e.g. cv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const ITensorDataPitchDevice &ksize, const ITensorDataPitchDevice &scale, NVCVBorderType borderMode,
                    cudaStream_t stream);
};

class EraseVarShape : public CudaBaseOp
{
public:
    EraseVarShape() = delete;

    EraseVarShape(DataShape max_input_shape, DataShape max_output_shape, int num_erasing_area);

    ~EraseVarShape();

    /**
    * @brief erase areas of images. Different images in the same batch can be erased differently.
    * @param inbatch gpu pointer, inputs[0] are batched input images, whose shape is input_shape and type is data_type.
    * @param outbatch gpu pointer, outputs[0] are batched output images that have the same type as data_type.
    * @param anchor an array of size num_erasing_area that gives the x coordinate and y coordinate of the top left point in the eraseing areas.
    * @param erasing an array of size num_erasing_area that gives the widths of the eraseing areas, the heights of the eraseing areas and
    *               integers in range 0-15, each of whose bits indicates whether or not the corresponding channel need to be erased.
    * @param values an array of size num_erasing_area*4 that gives the filling value for each erase area.
    * @param imgIdx an array of size num_erasing_area that maps a erase area idx to img idx in the batch.
    * @param random an boolean for random op.
    * @param seed random seed for random filling erase area
    * @param inplace for perform inplace op.
    * @param stream for the asynchronous execution.
    */
    ErrorCode infer(const IImageBatchVarShape &inbatch, const IImageBatchVarShape &outbatch,
                    const ITensorDataPitchDevice &anchor, const ITensorDataPitchDevice &erasing,
                    const ITensorDataPitchDevice &values, const ITensorDataPitchDevice &imgIdx, bool random,
                    unsigned int seed, bool inplace, cudaStream_t stream);

protected:
    int3  *d_max_values;
    void  *temp_storage;
    size_t storage_bytes;
    int    max_num_erasing_area;
};

class GaussianVarShape : public CudaBaseOp
{
public:
    GaussianVarShape() = delete;

    GaussianVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize, int maxBatchSize);

    ~GaussianVarShape();

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blurs each image using a Gaussian filter.
     * @param inData Input images.
     * @param outData Output images.
     * @param kernelSize Gaussian kernel size.
     * @param sigma Gaussian kernel standard deviation in X and Y directions.
     *              If sigma.y is zero or negative, use sigma.y = sigma.x.
     * @param borderMode pixel extrapolation method, e.g. NVCV_BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const ITensorDataPitchDevice &kernelSize, const ITensorDataPitchDevice &sigma,
                    NVCVBorderType borderMode, cudaStream_t stream);

    /**
     * @brief calculate the gpu buffer size needed by this operator
     * @param maxKernelSize Maximum Gaussian kernel size that may be used
     * @param maxBatchSize Maximum batch size that may be used
     */
    size_t calBufferSize(Size2D maxKernelSize, int maxBatchSize);

private:
    Size2D m_maxKernelSize = {0, 0};
    int    m_maxBatchSize  = 0;
    float *m_kernel        = nullptr;
};

class AverageBlurVarShape : public CudaBaseOp
{
public:
    AverageBlurVarShape() = delete;

    AverageBlurVarShape(DataShape max_input_shape, DataShape max_output_shape, Size2D maxKernelSize, int maxBatchSize);

    ~AverageBlurVarShape();

    /**
     * Limitations:
     *
     * Input:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Output:
     *      Data Layout:    [kNHWC, kHWC]
     *      Channels:       [1, 3, 4]
     *
     *      Data Type      | Allowed
     *      -------------- | -------------
     *      8bit  Unsigned | Yes
     *      8bit  Signed   | No
     *      16bit Unsigned | Yes
     *      16bit Signed   | Yes
     *      32bit Unsigned | No
     *      32bit Signed   | Yes
     *      32bit Float    | Yes
     *      64bit Float    | No
     *
     * Input/Output dependency
     *
     *      Property      |  Input == Output
     *     -------------- | -------------
     *      Data Layout   | Yes
     *      Data Type     | Yes
     *      Number        | Yes
     *      Channels      | Yes
     *      Width         | Yes
     *      Height        | Yes
     *
     * @brief Blur each image using an average filter.
     * @param inData Input images.
     * @param outData Output images.
     * @param kernelSize Average blur kernel size.
     *                     + Must be 1D tensor of int2, NVCV_PIXEL_TYPE_2S32
     * @param kernelAnchor Anchor of the kernel that indicates the relative position of a filtered point within the kernel.
     * (-1,-1) means that the anchor is at the kernel center.
     *                     + Must be 1D tensor of int2, NVCV_PIXEL_TYPE_2S32
     * @param borderMode pixel extrapolation method, e.g. cv::BORDER_CONSTANT
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &inData, const IImageBatchVarShapeDataPitchDevice &outData,
                    const ITensorDataPitchDevice &kernelSize, const ITensorDataPitchDevice &kernelAnchor,
                    NVCVBorderType borderMode, cudaStream_t stream);

    /**
     * @brief calculate the cpu/gpu buffer size needed by this operator
     * @param maxKernelSize Maximum Gaussian kernel size that may be used
     * @param maxBatchSize Maximum batch size that may be used
     */
    size_t calBufferSize(Size2D maxKernelSize, int maxBatchSize);

private:
    Size2D m_maxKernelSize = {0, 0};
    int    m_maxBatchSize  = 0;
    float *m_kernel        = nullptr;
};

class MedianBlurVarShape : public CudaBaseOp
{
public:
    MedianBlurVarShape() = delete;
    MedianBlurVarShape(const int maxVarShapeBatchSize);

    ~MedianBlurVarShape();
    /**
     * @brief Blur an image using a median kernel.
     * @param inputs gpu pointer, inputs[i] is input image where i ranges from 0 to batch-1, whose shape is
     * input_shape[i] and type is data_type.
     * @param outputs gpu pointer, outputs[i] is output image where i ranges from 0 to batch-1, whose size is
     * input_shape[i] and type is data_type.
     * @param gpu_workspace gpu pointer, gpu memory used to store the temporary variable.
     * @param cpu_workspace cpu pointer, cpu memory used to store the temporary variable.
     * @param batch batch size of the input images.
     * @param buffer_size size of the gpu_workspace/cpu_workspace.
     * @param ksize median blur kernel size.
     * @param input_shapes shape of the input images.
     * @param format format of the input images, e.g. kNHWC.
     * @param data_type data type of the input images, e.g. kCV_32F.
     * @param stream for the asynchronous execution.
     */
    ErrorCode infer(const IImageBatchVarShapeDataPitchDevice &in, const IImageBatchVarShapeDataPitchDevice &out,
                    const ITensorDataPitchDevice &ksize, cudaStream_t stream);

protected:
    const int        m_maxBatchSize;
    std::vector<int> m_kernelSizes;
};

} // namespace nv::cv::legacy::cuda_op

#endif // CV_CUDA_LEGACY_H
