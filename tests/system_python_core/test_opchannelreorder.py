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
import numpy as np
import util


def test_op_channelreorder_varshape():

    input = util.create_image_batch(10, nvcv.Format.RGB8, size=(123, 321))
    order = util.create_tensor((10, 3), np.int32, "NC", max_random=(2, 2, 2))

    out = input.channelreorder(order)
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize

    order = util.create_tensor((10, 4), np.int32, "NC", max_random=(3, 3, 3, 3))
    out = input.channelreorder(order, format=nvcv.Format.BGRA8)

    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == nvcv.Format.BGRA8
    assert out.maxsize == input.maxsize

    stream = nvcv.cuda.Stream()
    out = util.clone_image_batch(input)
    tmp = input.channelreorder_into(output=out, orders=order, stream=stream)
    assert tmp is out
    assert len(out) == len(input)
    assert out.capacity == input.capacity
    assert out.uniqueformat == input.uniqueformat
    assert out.maxsize == input.maxsize
