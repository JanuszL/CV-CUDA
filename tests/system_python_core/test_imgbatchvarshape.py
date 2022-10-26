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


def test_imgbatchvarshape_creation_works():
    batch = nvcv.ImageBatchVarShape(15, nvcv.Format.RGBA8)
    assert batch.capacity == 15
    assert len(batch) == 0
    assert batch.format == nvcv.Format.RGBA8

    # range must be empty
    cnt = 0
    for i in batch:
        cnt += 1
    assert cnt == 0


def test_imgbatchvarshape_one_image():
    batch = nvcv.ImageBatchVarShape(15, nvcv.Format.RGBA8)

    img = nvcv.Image((64, 32), nvcv.Format.RGBA8)
    batch.pushback(img)
    assert len(batch) == 1

    # range must contain one
    cnt = 0
    for bimg in batch:
        assert bimg is img
        cnt += 1
    assert cnt == 1

    # remove added image
    batch.popback()

    # check if its indeed removed
    assert len(batch) == 0
    cnt = 0
    for bimg in batch:
        cnt += 1
    assert cnt == 0


def test_imgbatchvarshape_several_images():
    batch = nvcv.ImageBatchVarShape(15, nvcv.Format.RGBA8)

    # add 4 images with different dimensions
    imgs = [nvcv.Image((m * 2, m), nvcv.Format.RGBA8) for m in range(2, 10, 2)]
    batch.pushback(imgs)
    assert len(batch) == 4

    # check if they were really added
    cnt = 0
    for bimg in batch:
        assert bimg is imgs[cnt]
        cnt += 1
    assert cnt == 4

    # now remove the last 2
    batch.popback(2)
    assert len(batch) == 2
    cnt = 0
    for bimg in batch:
        cnt += 1
    assert cnt == 2

    # clear everything
    batch.clear()
    assert len(batch) == 0
    cnt = 0
    for bimg in batch:
        cnt += 1
    assert cnt == 0
