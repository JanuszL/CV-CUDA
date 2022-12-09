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
