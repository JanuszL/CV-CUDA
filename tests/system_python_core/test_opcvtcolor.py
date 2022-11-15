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
    "input, code, output",
    [
        (
            nvcv.Tensor(5, [16, 23], nvcv.Format.BGR8),
            nvcv.ColorConversion.BGR2RGB,
            nvcv.Tensor(5, [16, 23], nvcv.Format.RGB8),
        ),
        (
            nvcv.Tensor(3, [86, 22], nvcv.Format.RGBA8),
            nvcv.ColorConversion.RGBA2BGRA,
            nvcv.Tensor(3, [86, 22], nvcv.Format.BGRA8),
        ),
        (
            nvcv.Tensor(7, [13, 21], nvcv.Format.Y8),
            nvcv.ColorConversion.GRAY2BGR,
            nvcv.Tensor(7, [13, 21], nvcv.Format.BGR8),
        ),
        (
            nvcv.Tensor(9, [66, 99], nvcv.Format.HSV8),
            nvcv.ColorConversion.HSV2RGB,
            nvcv.Tensor(9, [66, 99], nvcv.Format.RGB8),
        ),
        (
            nvcv.Tensor([1, 61, 62, 3], np.uint8, "NHWC"),
            nvcv.ColorConversion.YUV2RGB,
            nvcv.Tensor([1, 61, 62, 3], np.uint8, "NHWC"),
        ),
    ],
)
def test_op_cvtcolor(input, code, output):
    out = nvcv.cvtcolor(input, code)
    assert out.shape == output.shape
    assert out.dtype == output.dtype

    stream = nvcv.cuda.Stream()
    tmp = nvcv.cvtcolor_into(
        src=input,
        dst=output,
        code=code,
        stream=stream,
    )
    assert tmp is output
    assert output.shape[:-1] == input.shape[:-1]


@t.mark.parametrize(
    "num_images, in_format, img_size, max_pixel, code, out_format",
    [
        (
            10,
            nvcv.Format.RGB8,
            (123, 321),
            256,
            nvcv.ColorConversion.RGB2RGBA,
            nvcv.Format.RGBA8,
        ),
        (
            8,
            nvcv.Format.BGRA8,
            (23, 21),
            256,
            nvcv.ColorConversion.BGRA2RGB,
            nvcv.Format.RGB8,
        ),
        (
            6,
            nvcv.Format.RGB8,
            (23, 21),
            256,
            nvcv.ColorConversion.RGB2GRAY,
            nvcv.Format.Y8_ER,
        ),
        (
            4,
            nvcv.Format.HSV8,
            (23, 21),
            256,
            nvcv.ColorConversion.HSV2RGB,
            nvcv.Format.RGB8,
        ),
        (
            2,
            nvcv.Format.Y8_ER,
            (23, 21),
            256,
            nvcv.ColorConversion.GRAY2BGR,
            nvcv.Format.BGR8,
        ),
    ],
)
def test_op_cvtcolorvarshape(
    num_images, in_format, img_size, max_pixel, code, out_format
):
    input = util.create_image_batch(
        num_images, in_format, size=img_size, max_random=max_pixel, rng=RNG
    )
    output = util.create_image_batch(
        num_images, out_format, size=img_size, max_random=max_pixel, rng=RNG
    )
    out = nvcv.cvtcolor(input, code)
    assert len(out) == len(output)
    assert out.capacity == output.capacity
    assert out.maxsize == output.maxsize

    stream = nvcv.cuda.Stream()
    tmp = nvcv.cvtcolor_into(
        src=input,
        dst=output,
        code=code,
        stream=stream,
    )
    assert tmp is output
    assert len(output) == len(input)
    assert output.capacity == input.capacity
    assert output.maxsize == input.maxsize
