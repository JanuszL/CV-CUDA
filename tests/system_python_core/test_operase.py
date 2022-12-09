# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nvcv
import pytest as t


@t.mark.parametrize(
    "input, erasing_area_num, random, seed",
    [
        (nvcv.Tensor([1, 460, 640, 3], nvcv.Type.U8, "NHWC"), 1, False, 0),
        (nvcv.Tensor([5, 460, 640, 3], nvcv.Type.U8, "NHWC"), 1, True, 1),
    ],
)
def test_op_erase(input, erasing_area_num, random, seed):

    parameter_shape = [erasing_area_num]
    anchor = nvcv.Tensor(parameter_shape, nvcv.Type._2S32, "N")
    erasing = nvcv.Tensor(parameter_shape, nvcv.Type._3S32, "N")
    imgIdx = nvcv.Tensor(parameter_shape, nvcv.Type.S32, "N")
    values = nvcv.Tensor(parameter_shape, nvcv.Type.F32, "N")

    out = input.erase(anchor, erasing, values, imgIdx)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.erase_into(out, anchor, erasing, values, imgIdx)
    assert tmp is out

    stream = nvcv.cuda.Stream()
    out = input.erase(
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = input.erase_into(
        out=out,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp is out
