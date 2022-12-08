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
    "input,base,scale,globalscale,globalshift,epsilon,flags",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            nvcv.Tensor([1, 1], np.float32, "HW"),
            nvcv.Tensor([1, 1], np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            nvcv.Tensor([16, 1], np.float32, "HW"),
            nvcv.Tensor([16, 1], np.float32, "HW"),
            1,
            2,
            3,
            nvcv.NormalizeFlags.SCALE_IS_STDDEV,
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            nvcv.Tensor([1, 23], np.float32, "HW"),
            nvcv.Tensor([1, 23], np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            nvcv.Tensor([16, 23], np.float32, "HW"),
            nvcv.Tensor([16, 23], np.float32, "HW"),
            1,
            2,
            3,
            None,
        ),
    ],
)
def test_op_normalize(input, base, scale, globalscale, globalshift, epsilon, flags):
    out = input.normalize(base, scale)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.normalize_into(out, base, scale)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = input.normalize(
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    tmp = input.normalize_into(
        out=out,
        base=base,
        scale=scale,
        flags=flags,
        globalscale=globalscale,
        globalshift=globalshift,
        epsilon=epsilon,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
