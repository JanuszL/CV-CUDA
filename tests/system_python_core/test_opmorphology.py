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
import util


@t.mark.parametrize(
    "input, morphologyType, maskSize, anchor, iteration, border ",
    [
        (
            nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"),
            nvcv.MorphologyType.ERODE,
            [-1, -1],
            [-1, -1],
            1,
            nvcv.Border.CONSTANT,
        ),
        (
            nvcv.Tensor([4, 4, 3], np.float32, "HWC"),
            nvcv.MorphologyType.DILATE,
            [2, 1],
            [-1, -1],
            1,
            nvcv.Border.REPLICATE,
        ),
        (
            nvcv.Tensor([3, 88, 13, 3], np.uint16, "NHWC"),
            nvcv.MorphologyType.ERODE,
            [2, 2],
            [-1, -1],
            2,
            nvcv.Border.REFLECT,
        ),
        (
            nvcv.Tensor([3, 4, 4], np.uint16, "HWC"),
            nvcv.MorphologyType.DILATE,
            [3, 3],
            [-1, -1],
            1,
            nvcv.Border.WRAP,
        ),
        (
            nvcv.Tensor([1, 2, 3, 4], np.uint8, "NHWC"),
            nvcv.MorphologyType.ERODE,
            [-1, -1],
            [1, 1],
            1,
            nvcv.Border.REFLECT101,
        ),
    ],
)
def test_op_morphology(input, morphologyType, maskSize, anchor, iteration, border):
    out = input.morphology(
        morphologyType, maskSize, anchor, iteration=iteration, border=border
    )
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = nvcv.Tensor(input.shape, input.dtype, input.layout)
    tmp = input.morphology_into(
        output=out,
        morphologyType=morphologyType,
        maskSize=maskSize,
        anchor=anchor,
        iteration=iteration,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == input.dtype


@t.mark.parametrize(
    "num_images, img_format, img_size, max_pixel, \
     morphologyType, max_mask, max_anchor, iteration, border ",
    [
        (
            10,
            nvcv.Format.RGB8,
            (123, 321),
            256,
            nvcv.MorphologyType.ERODE,
            3,
            1,
            1,
            nvcv.Border.CONSTANT,
        ),
        (
            7,
            nvcv.Format.RGBf32,
            (62, 35),
            1.0,
            nvcv.MorphologyType.DILATE,
            4,
            2,
            2,
            nvcv.Border.REPLICATE,
        ),
        (
            1,
            nvcv.Format.F32,
            (33, 48),
            1234,
            nvcv.MorphologyType.DILATE,
            5,
            1,
            3,
            nvcv.Border.REFLECT,
        ),
        (
            3,
            nvcv.Format.U8,
            (23, 18),
            123,
            nvcv.MorphologyType.DILATE,
            5,
            4,
            4,
            nvcv.Border.WRAP,
        ),
        (
            6,
            nvcv.Format.F32,
            (77, 42),
            123456,
            nvcv.MorphologyType.ERODE,
            6,
            3,
            1,
            nvcv.Border.REFLECT101,
        ),
    ],
)
def test_op_morphology_varshape(
    num_images,
    img_format,
    img_size,
    max_pixel,
    morphologyType,
    max_mask,
    max_anchor,
    iteration,
    border,
):

    input = util.create_image_batch(num_images, img_format, img_size, max_pixel)

    masks = util.create_tensor(
        (num_images, 2),
        np.int32,
        "NC",
        max_random=max_mask,
        odd_only=True,
    )

    anchors = util.create_tensor(
        (num_images, 2),
        np.int32,
        "NC",
        max_random=max_anchor,
        odd_only=True,
    )

    out = input.morphology(
        morphologyType, masks, anchors, iteration=iteration, border=border
    )

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()

    out = util.clone_image_batch(input)
    tmp = input.morphology_into(
        output=out,
        morphologyType=morphologyType,
        masks=masks,
        anchors=anchors,
        iteration=iteration,
        border=border,
        stream=stream,
    )
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
