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
from random import randint


@t.mark.parametrize(
    "num_images, format, min_out_size, max_out_size, border_mode, border_value",
    [
        (
            4,
            nvcv.RGBf32,
            (1, 1),
            (128, 128),
            nvcv.Border.CONSTANT,
            [0],
        ),
        (
            5,
            nvcv.RGB8,
            (1, 1),
            (128, 128),
            nvcv.Border.CONSTANT,
            [12, 3, 4, 55],
        ),
        (
            9,
            nvcv.RGBA8,
            (1, 1),
            (128, 128),
            nvcv.Border.WRAP,
            [0],
        ),
        (
            12,
            nvcv.RGBAf32,
            (1, 1),
            (128, 128),
            nvcv.Border.REPLICATE,
            [0],
        ),
        (
            8,
            nvcv.RGB8,
            (1, 1),
            (128, 128),
            nvcv.Border.REFLECT,
            [0],
        ),
        (
            10,
            nvcv.RGBA8,
            (1, 1),
            (128, 128),
            nvcv.Border.REFLECT101,
            [0],
        ),
    ],
)
def test_op_copymakeborder(
    num_images, format, min_out_size, max_out_size, border_mode, border_value
):
    stream = nvcv.cuda.Stream()
    max_out_w = randint(min_out_size[0], max_out_size[0])
    max_out_h = randint(min_out_size[1], max_out_size[1])

    input = nvcv.ImageBatchVarShape(num_images)
    varshape_out = nvcv.ImageBatchVarShape(num_images)
    out_heights = []
    out_widths = []
    for i in range(num_images):
        w = randint(1, max_out_w)
        h = randint(1, max_out_h)
        img_i = nvcv.Image([w, h], format)
        input.pushback(img_i)
        w_out = randint(w, max_out_size[0])
        h_out = randint(h, max_out_size[1])
        img_o = nvcv.Image([w_out, h_out], format)
        varshape_out.pushback(img_o)
        out_heights.append(h_out)
        out_widths.append(w_out)

    top_tensor = nvcv.Tensor([1, 1, num_images, 1], nvcv.Type.S32, "NHWC")
    left_tensor = nvcv.Tensor([1, 1, num_images, 1], nvcv.Type.S32, "NHWC")

    tensor_out = nvcv.Tensor(num_images, [max_out_w, max_out_h], format)

    out = input.copymakeborderstack(
        top=top_tensor, left=left_tensor, out_height=max_out_h, out_width=max_out_w
    )
    assert out.layout == tensor_out.layout
    assert out.shape == tensor_out.shape
    assert out.dtype == tensor_out.dtype

    tmp = input.copymakeborderstack_into(
        tensor_out,
        top=top_tensor,
        left=left_tensor,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is tensor_out

    out = input.copymakeborder(
        top=top_tensor, left=left_tensor, out_heights=out_heights, out_widths=out_widths
    )
    assert out.uniqueformat is not None
    assert out.uniqueformat == varshape_out.uniqueformat
    for res, ref in zip(out, varshape_out):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = input.copymakeborder_into(
        varshape_out,
        top=top_tensor,
        left=left_tensor,
        border_mode=border_mode,
        border_value=border_value,
        stream=stream,
    )
    assert tmp is varshape_out
