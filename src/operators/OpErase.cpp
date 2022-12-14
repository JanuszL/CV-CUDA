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
#include <operators/OpErase.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpErase.hpp>
#include <util/Assert.h>

namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopEraseCreate, (NVCVOperatorHandle * handle, int32_t max_num_erasing_area))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv_op::Erase(max_num_erasing_area));
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopEraseSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                 NVCVTensorHandle anchor, NVCVTensorHandle erasing, NVCVTensorHandle values, NVCVTensorHandle imgIdx,
                 bool random, uint32_t seed))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle input(in), output(out), anchorwrap(anchor), erasingwrap(erasing),
                valueswrap(values), imgIdxwrap(imgIdx);
            priv::ToDynamicRef<priv_op::Erase>(handle)(stream, input, output, anchorwrap, erasingwrap, valueswrap,
                                                       imgIdxwrap, random, seed);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopEraseVarShapeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                 NVCVTensorHandle anchor, NVCVTensorHandle erasing, NVCVTensorHandle values, NVCVTensorHandle imgIdx,
                 bool random, uint32_t seed))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::ImageBatchVarShapeWrapHandle input(in), output(out);
            nv::cv::TensorWrapHandle anchorwrap(anchor), erasingwrap(erasing), valueswrap(values), imgIdxwrap(imgIdx);
            priv::ToDynamicRef<priv_op::Erase>(handle)(stream, input, output, anchorwrap, erasingwrap, valueswrap,
                                                       imgIdxwrap, random, seed);
        });
}
