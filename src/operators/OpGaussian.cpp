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
#include <operators/OpGaussian.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpGaussian.hpp>
#include <util/Assert.h>

namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopGaussianCreate,
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
                new priv_op::Gaussian(nv::cv::Size2D{maxKernelWidth, maxKernelHeight}, maxVarShapeBatchSize));
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopGaussianSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                 int32_t kernelWidth, int32_t kernelHeight, double sigmaX, double sigmaY, NVCVBorderType borderMode))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle output(out), input(in);
            priv::ToDynamicRef<priv_op::Gaussian>(handle)(
                stream, input, output, nv::cv::Size2D{kernelWidth, kernelHeight}, double2{sigmaX, sigmaY}, borderMode);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopGaussianVarShapeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                 NVCVTensorHandle kernelSize, NVCVTensorHandle sigma, NVCVBorderType borderMode))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::ImageBatchVarShapeWrapHandle inWrap(in), outWrap(out);
            nv::cv::TensorWrapHandle             kernelSizeWrap(kernelSize), sigmaWrap(sigma);
            priv::ToDynamicRef<priv_op::Gaussian>(handle)(stream, inWrap, outWrap, kernelSizeWrap, sigmaWrap,
                                                          borderMode);
        });
}
