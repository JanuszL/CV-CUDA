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
    out_shape = [i for i in input.shape]
    cdim = len(out_shape) - 1
    out_shape[cdim - 2] += top + bottom
    out_shape[cdim - 1] += left + right
    out = input.copymakeborder(top=top, bottom=bottom, left=left, right=right)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = nvcv.cuda.Stream()
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
