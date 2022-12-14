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
    "input, diameter, sigma_color, sigma_space, border",
    [
        (
            nvcv.Tensor([5, 9, 9, 4], np.uint8, "NHWC"),
            9,
            1,
            1,
            nvcv.Border.CONSTANT,
        ),
        (
            nvcv.Tensor([9, 9, 3], np.uint8, "HWC"),
            7,
            3,
            10,
            nvcv.Border.WRAP,
        ),
        (
            nvcv.Tensor([5, 21, 21, 4], np.uint8, "NHWC"),
            6,
            15,
            9,
            nvcv.Border.REPLICATE,
        ),
        (
            nvcv.Tensor([21, 21, 3], np.uint8, "HWC"),
            12,
            2,
            5,
            nvcv.Border.REFLECT,
        ),
    ],
)
def test_op_bilateral_filter(input, diameter, sigma_color, sigma_space, border):
    out = input.bilateral_filter(diameter, sigma_color, sigma_space, border)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.bilateral_filter_into(
        output=out,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
