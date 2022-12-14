/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <operators/OpAverageBlur.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpAverageBlur.hpp>
#include <util/Assert.h>

namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopAverageBlurCreate,
                (NVCVOperatorHandle * handle, int32_t maxKernelWidth, int32_t maxKernelHeight,
                 int32_t maxVarShapeBatchSize))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new priv_op::AverageBlur(nv::cv::Size2D{maxKernelWidth, maxKernelHeight}, maxVarShapeBatchSize));
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopAverageBlurSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                 int32_t kernelWidth, int32_t kernelHeight, int32_t kernelAnchorX, int32_t kernelAnchorY,
                 NVCVBorderType borderMode))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle output(out), input(in);
            priv::ToDynamicRef<priv_op::AverageBlur>(handle)(stream, input, output,
                                                             nv::cv::Size2D{kernelWidth, kernelHeight},
                                                             int2{kernelAnchorX, kernelAnchorY}, borderMode);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopAverageBlurVarShapeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                 NVCVTensorHandle kernelSize, NVCVTensorHandle kernelAnchor, NVCVBorderType borderMode))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::ImageBatchVarShapeWrapHandle inWrap(in), outWrap(out);
            nv::cv::TensorWrapHandle             kernelSizeWrap(kernelSize), kernelAnchorWrap(kernelAnchor);
            priv::ToDynamicRef<priv_op::AverageBlur>(handle)(stream, inWrap, outWrap, kernelSizeWrap, kernelAnchorWrap,
                                                             borderMode);
        });
}
