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
import pytest as t
import numpy as np
import util


RNG = np.random.default_rng(0)


@t.mark.parametrize(
    "input, flip_code",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            0,
        ),
        (
            nvcv.Tensor([4, 4, 3], np.float32, "HWC"),
            1,
        ),
        (
            nvcv.Tensor([3, 88, 13, 3], np.uint16, "NHWC"),
            -1,
        ),
        (
            nvcv.Tensor([3, 4, 4], np.int32, "HWC"),
            1,
        ),
        (
            nvcv.Tensor([1, 2, 3, 4], np.uint16, "NHWC"),
            0,
        ),
    ],
)
def test_op_flip(input, flip_code):
    out = input.flip(flip_code)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.flip_into(
        output=out,
        flipCode=flip_code,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, flip_code",
    [
        (
            10,
            nvcv.Format.RGB8,
            (123, 321),
            256,
            1,
        ),
        (
            7,
            nvcv.Format.RGBf32,
            (62, 35),
            1.0,
            1,
        ),
        (
            1,
            nvcv.Format.U16,
            (33, 48),
            1234,
            1,
        ),
        (
            13,
            nvcv.Format.U16,
            (26, 52),
            1234,
            1,
        ),
        (
            6,
            nvcv.Format.S32,
            (77, 42),
            123456,
            1,
        ),
    ],
)
def test_op_flipvarshape(num_images, img_format, img_size, max_pixel, flip_code):

    input = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    flipCode = util.create_tensor(
        (num_images, 1), np.int32, "NC", max_random=flip_code, rng=RNG
    )

    out = input.flip(flipCode)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = util.clone_image_batch(input)
    tmp = input.flip_into(
        output=out,
        flipCode=flipCode,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize