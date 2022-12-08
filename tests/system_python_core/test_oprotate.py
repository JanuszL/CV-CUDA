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
    "input, angle_deg, shift, interpolation",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            30,
            [3, 4],
            nvcv.Interp.NEAREST,
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            60,
            [3, 4],
            nvcv.Interp.LINEAR,
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            90,
            [3, 4],
            nvcv.Interp.CUBIC,
        ),
        (
            nvcv.Tensor([7, 12, 3], np.uint8, "HWC"),
            30,
            [2, 3],
            nvcv.Interp.NEAREST,
        ),
        (
            nvcv.Tensor([7, 12, 3], np.uint8, "HWC"),
            60,
            [2, 3],
            nvcv.Interp.LINEAR,
        ),
        (
            nvcv.Tensor([7, 12, 3], np.uint8, "HWC"),
            90,
            [2, 3],
            nvcv.Interp.CUBIC,
        ),
    ],
)
def test_op_rotate(input, angle_deg, shift, interpolation):
    out = input.rotate(angle_deg, shift, interpolation)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()

    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.rotate_into(
        output=out,
        angle_deg=angle_deg,
        shift=shift,
        interpolation=interpolation,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "nimages, format, max_size, max_pixel, max_angle_deg, max_shift, interpolation",
    [
        (
            5,
            nvcv.Format.RGB8,
            (16, 23),
            128.0,
            180,
            [5, 5],
            nvcv.Interp.NEAREST,
        ),
        (
            5,
            nvcv.Format.RGB8,
            (16, 23),
            256.0,
            180,
            [5, 5],
            nvcv.Interp.LINEAR,
        ),
        (
            5,
            nvcv.Format.RGB8,
            (16, 23),
            256.0,
            180,
            [5, 5],
            nvcv.Interp.CUBIC,
        ),
    ],
)
def test_op_rotatevarshape(
    nimages,
    format,
    max_size,
    max_pixel,
    max_angle_deg,
    max_shift,
    interpolation,
):

    input = util.create_image_batch(
        nimages, format, max_size=max_size, max_random=max_pixel
    )

    angle_deg = util.create_tensor(
        (nimages, 1),
        np.float64,
        "NC",
        max_random=max_angle_deg,
    )

    shift = util.create_tensor(
        (nimages, 2),
        np.float64,
        "NC",
        max_random=max_shift,
    )

    out = input.rotate(
        angle_deg,
        shift,
        interpolation,
    )
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()

    out = util.clone_image_batch(input)
    tmp = input.rotate_into(
        output=out,
        angle_deg=angle_deg,
        shift=shift,
        interpolation=interpolation,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
