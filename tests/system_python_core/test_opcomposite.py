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
    "foreground, background, fgMask",
    [
        (
            nvcv.Tensor([5, 9, 9, 3], np.uint8, "NHWC"),
            nvcv.Tensor([5, 9, 9, 3], np.uint8, "NHWC"),
            nvcv.Tensor([5, 9, 9, 1], np.uint8, "NHWC"),
        ),
        (
            nvcv.Tensor([9, 9, 3], np.uint8, "HWC"),
            nvcv.Tensor([9, 9, 3], np.uint8, "HWC"),
            nvcv.Tensor([9, 9, 1], np.uint8, "HWC"),
        ),
        (
            nvcv.Tensor([5, 21, 10, 3], np.uint8, "NHWC"),
            nvcv.Tensor([5, 21, 10, 3], np.uint8, "NHWC"),
            nvcv.Tensor([5, 21, 10, 1], np.uint8, "NHWC"),
        ),
    ],
)
def test_op_composite(foreground, background, fgMask):
    out = foreground.composite(background, fgMask)
    assert out.layout == foreground.layout
    assert out.shape == foreground.shape
    assert out.dtype == foreground.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(foreground.shape, foreground.dtype, foreground.layout)
    tmp = foreground.composite_into(
        background=background,
        fgMask=fgMask,
        output=out,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == foreground.layout
    assert out.shape == foreground.shape
    assert out.dtype == foreground.dtype
