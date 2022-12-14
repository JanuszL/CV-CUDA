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


@t.mark.parametrize(
    "input, erasing_area_num, random, seed",
    [
        (nvcv.Tensor([1, 460, 640, 3], nvcv.Type.U8, "NHWC"), 1, False, 0),
        (nvcv.Tensor([5, 460, 640, 3], nvcv.Type.U8, "NHWC"), 1, True, 1),
    ],
)
def test_op_erase(input, erasing_area_num, random, seed):

    parameter_shape = [erasing_area_num]
    anchor = nvcv.Tensor(parameter_shape, nvcv.Type._2S32, "N")
    erasing = nvcv.Tensor(parameter_shape, nvcv.Type._3S32, "N")
    imgIdx = nvcv.Tensor(parameter_shape, nvcv.Type.S32, "N")
    values = nvcv.Tensor(parameter_shape, nvcv.Type.F32, "N")

    out = input.erase(anchor, erasing, values, imgIdx)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.erase_into(out, anchor, erasing, values, imgIdx)
    assert tmp is out

    stream = nvcv.cuda.Stream()
    out = input.erase(
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = input.erase_into(
        out=out,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp is out
