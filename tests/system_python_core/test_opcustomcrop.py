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
import cvcuda
import pytest as t
import numpy as np


@t.mark.parametrize(
    "input,rc,out_shape",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            nvcv.RectI(x=0, y=0, width=20, height=2),
            [5, 2, 20, 4],
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            nvcv.RectI(2, 13, 16, 2),
            [5, 2, 16, 4],
        ),
        (
            nvcv.Tensor([16, 23, 2], np.uint8, "HWC"),
            nvcv.RectI(2, 13, 16, 2),
            [2, 16, 2],
        ),
    ],
)
def test_op_customcrop(input, rc, out_shape):
    out = cvcuda.customcrop(input, rc)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.customcrop_into(out, input, rc)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    stream = nvcv.cuda.Stream()
    out = cvcuda.customcrop(input, rect=rc, stream=stream)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.customcrop_into(dst=out, src=input, rect=rc, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
