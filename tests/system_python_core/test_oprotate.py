# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvcv
import nvcv_operators  # noqa: F401
import pytest as t
import numpy as np
import util


RNG = np.random.default_rng(0)


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
        nimages, format, max_size=max_size, max_random=max_pixel, rng=RNG
    )

    angle_deg = util.create_tensor(
        (nimages), np.float64, "N", max_random=max_angle_deg, rng=RNG
    )

    shift = util.create_tensor(
        (nimages, 2), np.float64, "NC", max_random=max_shift, rng=RNG
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
