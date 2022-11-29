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
#include <operators/OpMorphology.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpMorphology.hpp>
#include <util/Assert.h>

namespace nvcv    = nv::cv;
namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvopMorphologyCreate, (NVCVOperatorHandle * handle))
{
    return nvcv::ProtectCall(
        [&]
        {
            if (handle == nullptr)
            {
                throw priv::Exception(NVCV_ERROR_INVALID_ARGUMENT, "Pointer to NVCVOperator handle must not be NULL");
            }

            *handle = reinterpret_cast<NVCVOperatorHandle>(new priv_op::Morphology());
        });
}

NVCV_DEFINE_API(0, 0, NVCVStatus, nvcvopMorphologySubmit,
                (NVCVOperatorHandle handle, cudaStream_t stream, NVCVTensorHandle in, NVCVTensorHandle out,
                 NVCVMorphologyType morphType, int maskWidth, int maskHeight, int anchorX, int anchorY, int iteration,
                 const NVCVBorderType borderMode))
{
    return nvcv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle input(in), output(out);
            nv::cv::Size2D           maskSize = {maskWidth, maskHeight};
            int2                     anchor   = {anchorX, anchorY};
            priv::ToDynamicRef<priv_op::Morphology>(handle)(stream, input, output, morphType, maskSize, anchor,
                                                            iteration, borderMode);
        });
}
