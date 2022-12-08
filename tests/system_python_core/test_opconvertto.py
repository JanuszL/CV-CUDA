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
    "input,dtype,scale,offset",
    [
        (nvcv.Tensor([5, 16, 23, 4], np.uint8, "NHWC"), np.float32, 1.2, 10.2),
        (nvcv.Tensor([16, 23, 2], np.uint8, "HWC"), np.int32, -1.2, -5.5),
    ],
)
def test_op_convertto(input, dtype, scale, offset):
    out = input.convertto(dtype)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = nvcv.Tensor(input.shape, dtype, input.layout)
    tmp = input.convertto_into(out)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    out = input.convertto(dtype, scale)
    out = input.convertto(dtype, scale, offset)

    out = nvcv.Tensor(input.shape, dtype, input.layout)
    tmp = input.convertto_into(out, scale)
    tmp = input.convertto_into(out, scale, offset)

    nvcv.cuda.Stream.default.sync()  # HACK WAR CVCUDA-344 bug
    stream = nvcv.cuda.Stream()
    out = input.convertto(dtype=dtype, scale=scale, offset=offset, stream=stream)
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype

    tmp = input.convertto_into(out=out, scale=scale, offset=offset, stream=stream)
    assert tmp is out
    assert out.layout == input.layout
    assert out.shape == input.shape
    assert out.dtype == dtype
