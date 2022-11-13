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


@t.mark.parametrize(
    "input,out_shape,interp",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            [5, 132, 15, 4],
            nvcv.Interp.LINEAR,
        ),
        (nvcv.Tensor([16, 23, 4], np.uint8, "HWC"), [132, 15, 4], nvcv.Interp.CUBIC),
        (nvcv.Tensor([16, 23, 1], np.uint8, "HWC"), [132, 15, 1], None),
    ],
)
def test_op_resize(input, out_shape, interp):
    if interp is None:
        out = input.resize(out_shape)
    else:
        out = input.resize(out_shape, interp)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(out_shape, input.dtype, input.layout)
    if interp is None:
        tmp = input.resize_into(out)
    else:
        tmp = input.resize_into(out, interp)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    stream = nvcv.cuda.Stream()
    if interp is None:
        out = input.resize(shape=out_shape, stream=stream)
    else:
        out = input.resize(shape=out_shape, interp=interp, stream=stream)
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype

    if interp is None:
        tmp = input.resize_into(out=out, stream=stream)
    else:
        tmp = input.resize_into(out=out, interp=interp, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == out_shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "inSize, outSize, interp",
    [((123, 321), (321, 123), nvcv.Interp.LINEAR), ((123, 321), (321, 123), None)],
)
def test_op_resizevarshape(inSize, outSize, interp):

    RNG = np.random.default_rng(0)

    input = util.create_image_batch(
        10, nvcv.Format.RGBA8, size=inSize, max_random=256, rng=RNG
    )

    base_output = util.create_image_batch(
        10, nvcv.Format.RGBA8, size=outSize, max_random=256, rng=RNG
    )

    sizes = []
    for image in base_output:
        sizes.append([image.width, image.height])

    if interp is None:
        out = input.resize(sizes)
    else:
        out = input.resize(sizes=sizes, interp=interp)

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == outSize

    stream = nvcv.cuda.Stream()
    if interp is None:
        tmp = input.resize_into(out=base_output, stream=stream)
    else:
        tmp = input.resize_into(out=base_output, interp=interp, stream=stream)
    assert tmp is base_output
    assert len(base_output) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == outSize
