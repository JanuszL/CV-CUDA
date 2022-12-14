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
    "input, kernel_size, sigma, border",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            [3, 3],
            [0.5, 0.5],
            nvcv.Border.CONSTANT,
        ),
        (
            nvcv.Tensor([4, 4, 3], np.float32, "HWC"),
            [5, 5],
            [0.8, 0.8],
            nvcv.Border.REPLICATE,
        ),
        (
            nvcv.Tensor([3, 88, 13, 3], np.uint16, "NHWC"),
            [7, 7],
            [0.7, 0.7],
            nvcv.Border.REFLECT,
        ),
        (
            nvcv.Tensor([3, 4, 4], np.int32, "HWC"),
            [9, 9],
            [0.8, 0.8],
            nvcv.Border.WRAP,
        ),
        (
            nvcv.Tensor([1, 2, 3, 4], np.int16, "NHWC"),
            [7, 7],
            [0.6, 0.6],
            nvcv.Border.REFLECT101,
        ),
    ],
)
def test_op_gaussian(input, kernel_size, sigma, border):
    out = input.gaussian(kernel_size, sigma, border)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.gaussian_into(
        output=out,
        kernel_size=kernel_size,
        sigma=sigma,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
