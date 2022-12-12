/* Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <operators/OpWarpAffine.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpWarpAffine.hpp>
#include <util/Assert.h>

namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopWarpAffineCreate,
                (NVCVOperatorHandle * handle, const int32_t maxVarShapeBatchSize))
{
    return priv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv_op::WarpAffine(maxVarShapeBatchSize));
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopWarpAffineSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                 const NVCVAffineTransform xform, const int32_t flags, const NVCVBorderType borderMode,
                 const float4 borderValue))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle input(in), output(out);
            priv::ToDynamicRef<priv_op::WarpAffine>(handle)(stream, input, output, xform, flags, borderMode,
                                                            borderValue);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopWarpAffineVarShapeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVImageBatchHandle out,
                 NVCVTensorHandle transMatrix, const int32_t flags, const NVCVBorderType borderMode,
                 const float4 borderValue))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::ImageBatchVarShapeWrapHandle input(in), output(out);
            nv::cv::TensorWrapHandle             transMatrixWrap(transMatrix);
            priv::ToDynamicRef<priv_op::WarpAffine>(handle)(stream, input, output, transMatrixWrap, flags, borderMode,
                                                            borderValue);
        });
}
