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


@t.mark.parametrize(
    "input,out_shape,interp,fmt",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            [5, 132, 15, 4],
            nvcv.Interp.LINEAR,
            nvcv.Format.RGB8,
        ),
        (
            nvcv.Tensor([5, 31, 31, 4], np.uint8, "NHWC"),
            [5, 55, 55, 4],
            nvcv.Interp.LINEAR,
            nvcv.Format.RGB8,
        ),
        (
            nvcv.Tensor([5, 55, 55, 4], np.uint8, "NHWC"),
            [5, 31, 31, 4],
            nvcv.Interp.LINEAR,
            nvcv.Format.RGB8,
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.float32, "NHWC"),
            [5, 132, 15, 4],
            nvcv.Interp.LINEAR,
            nvcv.Format.RGBf32,
        ),
        (
            nvcv.Tensor([5, 31, 31, 4], np.float32, "NHWC"),
            [5, 55, 55, 4],
            nvcv.Interp.LINEAR,
            nvcv.Format.RGBf32,
        ),
        (
            nvcv.Tensor([5, 55, 55, 4], np.float32, "NHWC"),
            [5, 31, 31, 4],
            nvcv.Interp.LINEAR,
            nvcv.Format.RGBf32,
        ),
    ],
)
def test_op_pillowresize(input, out_shape, interp, fmt):

    out = input.pillowresize(out_shape, fmt, interp)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(out_shape, input.dtype, input.layout)
    tmp = input.pillowresize_into(out, fmt, interp)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()
    stream = nvcv.cuda.Stream()
    tmp = input.pillowresize_into(
        out=out,
        format=fmt,
        interp=interp,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype
