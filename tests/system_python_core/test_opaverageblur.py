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
    "input, kernel_size, kernel_anchor, border",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            [3, 3],
            [1, 1],
            nvcv.Border.CONSTANT,
        ),
        (
            nvcv.Tensor([4, 4, 3], np.float32, "HWC"),
            [5, 5],
            [0, 0],
            nvcv.Border.REPLICATE,
        ),
        (
            nvcv.Tensor([3, 88, 13, 3], np.uint16, "NHWC"),
            [7, 7],
            [2, 2],
            nvcv.Border.REFLECT,
        ),
        (
            nvcv.Tensor([3, 4, 4], np.int32, "HWC"),
            [9, 9],
            [-1, -1],
            nvcv.Border.WRAP,
        ),
        (
            nvcv.Tensor([1, 2, 3, 4], np.int16, "NHWC"),
            [11, 11],
            [8, 8],
            nvcv.Border.REFLECT101,
        ),
    ],
)
def test_op_averageblur(input, kernel_size, kernel_anchor, border):
    out = input.averageblur(kernel_size, kernel_anchor, border)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.averageblur_into(
        output=out,
        kernel_size=kernel_size,
        kernel_anchor=kernel_anchor,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
