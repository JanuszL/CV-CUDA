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
import numpy as np
import util


RNG = np.random.default_rng(0)


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

    input = util.create_image_batch(
        num_images, img_format, size=img_size, max_random=max_pixel, rng=RNG
    )

    masks = util.create_tensor(
        (num_images, 2), np.int32, "NC", max_random=max_mask, rng=RNG
    )

    anchors = util.create_tensor(
        (num_images, 2), np.int32, "NC", max_random=max_anchor, rng=RNG
    )

    out = input.morphology(
        morphologyType, masks, anchors, iteration=iteration, border=border
    )

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

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
