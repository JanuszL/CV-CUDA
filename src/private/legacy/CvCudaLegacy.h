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
#include <nvcv/ITensorData.hpp>
#include <nvcv/Rect.h>

namespace nv::cv::legacy::cuda_op {

struct Size
{
    int width, height;

    __host__ __device__ Size(int _width, int _height)
        : width(_width)
        , height(_height)
    {
    }
};

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

} // namespace nv::cv::legacy::cuda_op

#endif // CV_CUDA_LEGACY_H
