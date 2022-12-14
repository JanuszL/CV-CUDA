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

#include <nvcv/Status.hpp>
#include <nvcv/Tensor.hpp>
#include <operators/OpConvertTo.hpp>
#include <private/core/Exception.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpConvertTo.hpp>
#include <util/Assert.h>

namespace nvcv    = nv::cv;
namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvopConvertToCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv_op::ConvertTo());
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopConvertToSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                 const double alpha, const double beta))
{
    return nvcv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle input(in), output(out);
            priv::ToDynamicRef<priv_op::ConvertTo>(handle)(stream, input, output, alpha, beta);
        });
}
