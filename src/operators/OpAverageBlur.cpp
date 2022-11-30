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
#include <operators/OpAverageBlur.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpAverageBlur.hpp>
#include <util/Assert.h>

namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopAverageBlurCreate,
                (NVCVOperatorHandle * handle, int maxKernelWidth, int maxKernelHeight, int maxVarShapeBatchSize))
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
                 int kernelWidth, int kernelHeight, int kernelAnchorX, int kernelAnchorY, NVCVBorderType borderMode))
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
