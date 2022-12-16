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
#include <operators/OpPillowResize.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpPillowResize.hpp>
#include <util/Assert.h>

namespace nvcv    = nv::cv;
namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopPillowResizeCreate,
                (NVCVOperatorHandle * handle, int32_t maxWidth, int32_t maxHeight, int32_t maxBatchSize,
                 NVCVImageFormat fmt))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }
            *handle = reinterpret_cast<NVCVOperatorHandle>(
                new priv_op::PillowResize(nvcv::Size2D{maxWidth, maxHeight}, maxBatchSize, fmt));
        });
}

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopPillowResizeSubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                 const NVCVInterpolationType interpolation))
{
    return nvcv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle input(in), output(out);
            priv::ToDynamicRef<priv_op::PillowResize>(handle)(stream, input, output, interpolation);
        });
}
