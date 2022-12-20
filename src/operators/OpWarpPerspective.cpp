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

#include "priv/OpWarpPerspective.hpp"

#include "priv/SymbolVersioning.hpp"

#include <nvcv/Exception.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <util/Assert.h>

namespace nvcv = nv::cv;
namespace priv = nv::cvop::priv;

NVCV_OP_DEFINE_API(0, 2, NVCVStatus, nvcvopWarpPerspectiveCreate,
                   (NVCVOperatorHandle * handle, const int maxVarShapeBatchSize))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw nvcv::Exception(nvcv::Status::ERROR_INVALID_ARGUMENT,
                                      "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv::WarpPerspective(maxVarShapeBatchSize));
        });
}

NVCV_OP_DEFINE_API(0, 2, NVCVStatus, nvcvopWarpPerspectiveSubmit,
                   (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                    const NVCVPerspectiveTransform transMatrix, const int flags, const NVCVBorderType borderMode,
                    const float4 borderValue))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::TensorWrapHandle input(in), output(out);
            priv::ToDynamicRef<priv::WarpPerspective>(handle)(stream, input, output, transMatrix, flags, borderMode,
                                                              borderValue);
        });
}

NVCV_OP_DEFINE_API(0, 2, NVCVStatus, nvcvopWarpPerspectiveVarShapeSubmit,
                   (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                    NVCVTensorHandle transMatrix, const int flags, const NVCVBorderType borderMode,
                    const float4 borderValue))
{
    return nvcv::ProtectCall(
        [&]
        {
            nvcv::ImageBatchVarShapeWrapHandle input(in), output(out);
            nvcv::TensorWrapHandle             transMatrixWrap(transMatrix);
            priv::ToDynamicRef<priv::WarpPerspective>(handle)(stream, input, output, transMatrixWrap, flags, borderMode,
                                                              borderValue);
        });
}
