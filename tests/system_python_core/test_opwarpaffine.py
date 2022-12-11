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
    "input, xform, flags, border_mode, border_value",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array([[1, 0, 0], [0, 1, 0]]),
            nvcv.Interp.NEAREST,
            nvcv.Border.CONSTANT,
            [],
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array([[1, 0, 0], [0, 1, 0]]),
            nvcv.Interp.NEAREST,
            nvcv.Border.CONSTANT,
            [0],
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array([[1, 2, 0], [2, 1, 1]]),
            nvcv.Interp.LINEAR,
            nvcv.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            np.array([[1, 2, 0], [2, 1, 1]]),
            nvcv.Interp.LINEAR,
            nvcv.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
        (
            nvcv.Tensor([11, 21, 4], np.uint8, "HWC"),
            np.array([[2, 2, 0], [3, 1, 0]]),
            nvcv.Interp.NEAREST,
            nvcv.Border.CONSTANT,
            [0],
        ),
        (
            nvcv.Tensor([11, 21, 4], np.uint8, "HWC"),
            np.array([[2, 2, 1], [3, 1, 2]]),
            nvcv.Interp.LINEAR,
            nvcv.Border.WRAP,
            [1, 2, 3, 4],
        ),
        (
            nvcv.Tensor([11, 21, 4], np.uint8, "HWC"),
            np.array([[1, 2, 0], [2, 1, 1]]),
            nvcv.Interp.LINEAR,
            nvcv.Border.REPLICATE,
            [1, 2, 3, 4],
        ),
    ],
)
def test_op_warp_affine(input, xform, flags, border_mode, border_value):
    out = input.warp_affine(
        xform, flags, border_mode=border_mode, border_value=border_value
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.warp_affine_into(
        output=out,
        xform=xform,
        flags=flags,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_xval, flags, bmode, border_value",
    [
        (
            5,
            nvcv.Format.RGB8,
            (16, 23),
            128.0,
            7,
            nvcv.Interp.NEAREST,
            nvcv.Border.CONSTANT,
            [],
        ),
        (
            5,
            nvcv.Format.RGB8,
            (16, 23),
            128.0,
            7,
            nvcv.Interp.NEAREST,
            nvcv.Border.CONSTANT,
            [1, 2, 3, 4],
        ),
        (
            4,
            nvcv.Format.RGB8,
            (16, 23),
            128.0,
            5,
            nvcv.Interp.LINEAR,
            nvcv.Border.WRAP,
            [0],
        ),
        (
            3,
            nvcv.Format.RGB8,
            (16, 23),
            128.0,
            4,
            nvcv.Interp.CUBIC,
            nvcv.Border.REPLICATE,
            [2, 1, 0],
        ),
    ],
)
def test_op_warp_affinevarshape(
    nimages, format, max_size, max_pixel, max_xval, flags, bmode, border_value
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel
    )

    xform = util.create_tensor((nimages, 6), np.float32, "NC", max_xval)

    out = input.warp_affine(xform, flags, border_mode=bmode, border_value=border_value)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()

    out = util.clone_image_batch(input)
    tmp = input.warp_affine_into(
        output=out,
        xform=xform,
        flags=flags,
        border_mode=bmode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
