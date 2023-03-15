# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cvcuda
import pytest as t


@t.mark.parametrize(
    "input, thtype",
    [
        (
            cvcuda.Tensor((1, 460, 640, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.BINARY,
        ),
        (
            cvcuda.Tensor((5, 640, 460, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.BINARY_INV,
        ),
        (
            cvcuda.Tensor((4, 1920, 1080, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TRUNC,
        ),
        (
            cvcuda.Tensor((2, 1000, 1000, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TOZERO,
        ),
        (
            cvcuda.Tensor((3, 100, 100, 3), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TOZERO_INV,
        ),
        (
            cvcuda.Tensor((5, 460, 640, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.OTSU | cvcuda.ThresholdType.BINARY,
        ),
        (
            cvcuda.Tensor((1, 1000, 1000, 1), cvcuda.Type.U8, "NHWC"),
            cvcuda.ThresholdType.TRIANGLE | cvcuda.ThresholdType.BINARY_INV,
        ),
    ],
)
def test_op_threshold(input, thtype):

    parameter_shape = (input.shape[0],)
    thresh = cvcuda.Tensor(parameter_shape, cvcuda.Type.F64, "N")
    maxval = cvcuda.Tensor(parameter_shape, cvcuda.Type.F64, "N")

    out = cvcuda.threshold(input, thresh, maxval, thtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = cvcuda.Tensor(input.shape, input.dtype, input.layout)
    tmp = cvcuda.threshold_into(out, input, thresh, maxval, thtype)
    assert tmp is out

    stream = cvcuda.Stream()
    out = cvcuda.threshold(
        src=input,
        thresh=thresh,
        maxval=maxval,
        type=thtype,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = cvcuda.threshold_into(
        src=input,
        dst=out,
        thresh=thresh,
        maxval=maxval,
        type=thtype,
        stream=stream,
    )
    assert tmp is out
