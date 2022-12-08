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
import numpy as np


@t.mark.parametrize(
    "input, top, bottom, left, right, border_mode, border_value",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            1,
            2,
            3,
            4,
            nvcv.Border.CONSTANT,
            [0],
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            1,
            2,
            3,
            4,
            nvcv.Border.CONSTANT,
            [12, 3, 4, 55],
        ),
        (nvcv.Tensor([16, 23, 4], np.uint8, "HWC"), 2, 2, 2, 2, nvcv.Border.WRAP, [0]),
        (
            nvcv.Tensor([16, 23, 3], np.uint8, "HWC"),
            10,
            12,
            35,
            18,
            nvcv.Border.REPLICATE,
            [0],
        ),
        (
            nvcv.Tensor([16, 23, 1], np.float32, "HWC"),
            11,
            1,
            20,
            3,
            nvcv.Border.REFLECT,
            [0],
        ),
        (
            nvcv.Tensor([16, 23, 3], np.float32, "HWC"),
            11,
            1,
            20,
            3,
            nvcv.Border.REFLECT101,
            [0],
        ),
    ],
)
def test_op_copymakeborder(input, top, bottom, left, right, border_mode, border_value):

    stream = nvcv.cuda.Stream()
    out_shape = [i for i in input.shape]
    cdim = len(out_shape) - 1
    out_shape[cdim - 2] += top + bottom
    out_shape[cdim - 1] += left + right
    out = input.copymakeborder(top=top, bottom=bottom, left=left, right=right)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(out_shape, input.dtype, input.layout)
    tmp = input.copymakeborder_into(
        output=out,
        top=top,
        left=left,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype
