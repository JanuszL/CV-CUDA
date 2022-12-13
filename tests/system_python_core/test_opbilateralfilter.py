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
