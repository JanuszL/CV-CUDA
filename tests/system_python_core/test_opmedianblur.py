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
    "input, ksize",
    [
        (
            nvcv.Tensor([5, 9, 9, 4], np.uint8, "NHWC"),
            [5, 5],
        ),
        (
            nvcv.Tensor([9, 9, 3], np.uint8, "HWC"),
            [5, 5],
        ),
        (
            nvcv.Tensor([5, 21, 21, 4], np.uint8, "NHWC"),
            [15, 15],
        ),
        (
            nvcv.Tensor([21, 21, 3], np.uint8, "HWC"),
            [15, 15],
        ),
    ],
)
def test_op_median_blur(input, ksize):
    out = input.median_blur(ksize)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.median_blur_into(
        output=out,
        ksize=ksize,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype
