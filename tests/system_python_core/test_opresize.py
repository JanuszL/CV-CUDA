# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import nvcv
import pytest as t
import numpy as np


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

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
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
