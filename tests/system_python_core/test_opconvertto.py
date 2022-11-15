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


@t.mark.parametrize(
    "input,dtype,scale,offset",
    [
        (nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"), np.float32, 1.2, 10.2),
        (nvcv.Tensor([16, 23, 2], np.uint8, "HWC"), np.int32, -1.2, -5.5),
    ],
)
def test_op_convertto(input, dtype, scale, offset):
    out = nvcv.convertto(input, dtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = nvcv.Tensor(input.shape, dtype, input.layout)
    tmp = nvcv.convertto_into(out, input)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = nvcv.convertto(input, dtype, scale)
    out = nvcv.convertto(input, dtype, scale, offset)

    out = nvcv.Tensor(input.shape, dtype, input.layout)
    tmp = nvcv.convertto_into(out, input, scale)
    tmp = nvcv.convertto_into(out, input, scale, offset)

    stream = nvcv.cuda.Stream()
    out = nvcv.convertto(
        src=input, dtype=dtype, scale=scale, offset=offset, stream=stream
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    tmp = nvcv.convertto_into(
        dst=out, src=input, scale=scale, offset=offset, stream=stream
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype
