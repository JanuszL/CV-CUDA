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
#include <operators/OpNormalize.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpNormalize.hpp>
#include <util/Assert.h>

namespace nvcv    = nv::cv;
namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvopNormalizeCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv_op::Normalize());
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvopNormalizeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle base,
                 NVCVTensorHandle scale, NVCVTensorHandle out, float global_scale, float shift, float epsilon,
                 uint32_t flags))
{
    return nvcv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle inWrap(in), baseWrap(base), scaleWrap(scale), outWrap(out);
            priv::ToDynamicRef<priv_op::Normalize>(handle)(stream, inWrap, baseWrap, scaleWrap, outWrap, global_scale,
                                                           shift, epsilon, flags);
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopNormalizeVarShapeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVImageBatchHandle in, NVCVTensorHandle base,
                 NVCVTensorHandle scale, NVCVImageBatchHandle out, float global_scale, float shift, float epsilon,
                 uint32_t flags))
{
    return nvcv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle             baseWrap(base), scaleWrap(scale);
            nv::cv::ImageBatchVarShapeWrapHandle inWrap(in), outWrap(out);
            priv::ToDynamicRef<priv_op::Normalize>(handle)(stream, inWrap, baseWrap, scaleWrap, outWrap, global_scale,
                                                           shift, epsilon, flags);
        });
}
