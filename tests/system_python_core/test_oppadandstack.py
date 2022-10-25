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
    "format,num_images,min_size,max_size,border,bvalue,out_shape,out_layout,out_dtype",
    [
        (
            nvcv.Format.RGBA8,
            1,
            (10, 5),
            (10, 5),
            nvcv.Border.REPLICATE,
            0,
            [1, 5, 10, 4],
            "NHWC",
            np.uint8,
        ),
    ],
)
def test_op_padandstack(
    format,
    num_images,
    min_size,
    max_size,
    border,
    bvalue,
    out_shape,
    out_layout,
    out_dtype,
):

    input = nvcv.ImageBatchVarShape(num_images, format)

    input.pushback(
        [
            nvcv.Image(
                (
                    min_size[0] + (max_size[0] - min_size[0]) * i // num_images,
                    min_size[1] + (max_size[1] - min_size[1]) * i // num_images,
                ),
                format,
            )
            for i in range(num_images)
        ]
    )

    left = nvcv.Tensor([1, 1, num_images, 1], np.int32, "NHWC")
    top = nvcv.Tensor([1, 1, num_images, 1], np.int32, "NHWC")

    out = input.padandstack(top, left)
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == out_dtype

    out = nvcv.Tensor(out_shape, out_dtype, out_layout)
    tmp = input.padandstack_into(out, top, left)
    assert tmp is out

    stream = nvcv.cuda.Stream()
    out = input.padandstack(
        left=left, top=top, border=border, bvalue=bvalue, stream=stream
    )
    assert out.layout == out_layout
    assert out.shape == out_shape
    assert out.dtype == out_dtype

    tmp = input.padandstack_into(
        out=out, left=left, top=top, border=border, bvalue=bvalue, stream=stream
    )
    assert tmp is out
