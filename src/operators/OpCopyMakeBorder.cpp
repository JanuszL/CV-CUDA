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
#include <operators/OpCopyMakeBorder.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpCopyMakeBorder.hpp>
#include <util/Assert.h>

namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopCopyMakeBorderCreate, (NVCVOperatorHandle * handle))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv_op::CopyMakeBorder());
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopCopyMakeBorderSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out, int32_t top,
                 int32_t left, NVCVBorderType borderMode, const float4 borderValue))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle output(out), input(in);
            priv::ToDynamicRef<priv_op::CopyMakeBorder>(handle)(stream, input, output, top, left, borderMode,
                                                                borderValue);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopCopyMakeBorderVarShapeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                 NVCVTensorHandle top, NVCVTensorHandle left, NVCVBorderType borderMode, const float4 borderValue))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::ImageBatchWrapHandle output(out), input(in);
            nv::cv::TensorWrapHandle     topVec(top), leftVec(left);
            priv::ToDynamicRef<priv_op::CopyMakeBorder>(handle)(stream, input, output, topVec, leftVec, borderMode,
                                                                borderValue);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopCopyMakeBorderVarShapeStackSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle out,
                 NVCVTensorHandle top, NVCVTensorHandle left, NVCVBorderType borderMode, const float4 borderValue))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::ImageBatchWrapHandle input(in);
            nv::cv::TensorWrapHandle     output(out), topVec(top), leftVec(left);
            priv::ToDynamicRef<priv_op::CopyMakeBorder>(handle)(stream, input, output, topVec, leftVec, borderMode,
                                                                borderValue);
        });
}
