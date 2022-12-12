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
import util


@t.mark.parametrize(
    "input, kernel, kernel_anchor, border",
    [
        (
            util.create_image_batch(
                10, nvcv.Format.RGB8, size=(123, 321), max_random=256
            ),
            util.create_image_batch(10, nvcv.Format.F32, size=(3, 3), max_random=1),
            util.create_tensor((10, 2), np.int32, "NC", max_random=(3, 3)),
            nvcv.Border.CONSTANT,
        ),
        (
            util.create_image_batch(7, nvcv.Format.RGBf32, max_random=1),
            util.create_image_batch(7, nvcv.Format.F32, size=(5, 5), max_random=3),
            util.create_tensor((7, 2), np.int32, "NC", max_random=(5, 5)),
            nvcv.Border.REPLICATE,
        ),
        (
            util.create_image_batch(1, nvcv.Format.U8, max_random=1234),
            util.create_image_batch(1, nvcv.Format.F32, size=(7, 7), max_random=2),
            util.create_tensor((1, 2), np.int32, "NC", max_random=(7, 7)),
            nvcv.Border.REFLECT,
        ),
        (
            util.create_image_batch(6, nvcv.Format.S16, max_random=1234),
            util.create_image_batch(6, nvcv.Format.F32, max_size=(9, 9), max_random=4),
            util.create_tensor((6, 2), np.int32, "NC", max_random=(1, 1)),
            nvcv.Border.WRAP,
        ),
        (
            util.create_image_batch(9, nvcv.Format.S32, max_random=12345),
            util.create_image_batch(9, nvcv.Format.F32, max_size=(4, 4), max_random=2),
            util.create_tensor((9, 2), np.int32, "NC", max_random=(4, 4)),
            nvcv.Border.REFLECT101,
        ),
    ],
)
def test_op_conv2dvarshape(input, kernel, kernel_anchor, border):
    out = input.conv2d(
        kernel,
        kernel_anchor,
        border,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = util.clone_image_batch(input)
    tmp = input.conv2d_into(
        output=out,
        kernel=kernel,
        kernel_anchor=kernel_anchor,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
