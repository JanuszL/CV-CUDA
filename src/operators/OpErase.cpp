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
#include <operators/OpErase.hpp>
#include <private/core/Exception.hpp>
#include <private/core/Status.hpp>
#include <private/core/SymbolVersioning.hpp>
#include <private/operators/OpErase.hpp>
#include <util/Assert.h>

namespace priv    = nv::cv::priv;
namespace priv_op = nv::cvop::priv;

NVCV_DEFINE_API(0, 2, NVCVStatus, nvcvopEraseCreate, (NVCVOperatorHandle * handle, int max_num_erasing_area))
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
                 NVCVTensorHandle anchor_x, NVCVTensorHandle anchor_y, NVCVTensorHandle erasing_w,
                 NVCVTensorHandle erasing_h, NVCVTensorHandle erasing_c, NVCVTensorHandle values,
                 NVCVTensorHandle imgIdx, bool random, unsigned int seed, bool inplace))
{
    return priv::ProtectCall(
        [&]
        {
            nv::cv::TensorWrapHandle input(in), output(out), anchorxwrap(anchor_x), anchorywrap(anchor_y),
                erasingwwrap(erasing_w), erasinghwrap(erasing_h), erasingcwrap(erasing_c), valueswrap(values),
                imgIdxwrap(imgIdx);
            priv::ToDynamicRef<priv_op::Erase>(handle)(stream, input, output, anchorxwrap, anchorywrap, erasingwwrap,
                                                       erasinghwrap, erasingcwrap, valueswrap, imgIdxwrap, random, seed,
                                                       inplace);
        });
}
