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
    "num_images, format, min_size, max_size, erasing_area_num, random, seed",
    [
        (1, nvcv.Format.U8, (100, 100), (200, 200), 1, False, 0),
        (5, nvcv.Format.RGB8, (100, 100), (200, 100), 1, True, 1),
    ],
)
def test_op_erase(
    num_images, format, min_size, max_size, erasing_area_num, random, seed
):

    parameter_shape = [erasing_area_num]
    anchor = nvcv.Tensor(parameter_shape, nvcv.Type._2S32, "N")
    erasing = nvcv.Tensor(parameter_shape, nvcv.Type._3S32, "N")
    imgIdx = nvcv.Tensor(parameter_shape, nvcv.Type.S32, "N")
    values = nvcv.Tensor(parameter_shape, nvcv.Type.F32, "N")

    input = nvcv.ImageBatchVarShape(num_images)
    output = nvcv.ImageBatchVarShape(num_images)
    for i in range(num_images):
        w = randint(min_size[0], max_size[0])
        h = randint(min_size[1], max_size[1])
        img_in = nvcv.Image([w, h], format)
        input.pushback(img_in)
        img_out = nvcv.Image([w, h], format)
        output.pushback(img_out)

    tmp = input.erase(anchor, erasing, values, imgIdx)
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = input.erase_into(
        output, anchor, erasing, values, imgIdx, random=random, seed=seed
    )
    assert tmp is output

    stream = nvcv.cuda.Stream()
    tmp = input.erase(
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp.uniqueformat is not None
    assert tmp.uniqueformat == output.uniqueformat
    for res, ref in zip(tmp, output):
        assert res.size == ref.size
        assert res.format == ref.format

    tmp = input.erase_into(
        out=output,
        anchor=anchor,
        erasing=erasing,
        values=values,
        imgIdx=imgIdx,
        random=random,
        seed=seed,
        stream=stream,
    )
    assert tmp is output
