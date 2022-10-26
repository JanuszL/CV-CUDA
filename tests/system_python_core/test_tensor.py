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
import numba
import numpy as np
from numba import cuda

assert numba.cuda.is_available()


@t.mark.parametrize(
    "n,size,fmt,gold_layout,gold_shape,gold_dtype",
    [
        (
            5,
            (32, 16),
            nvcv.Format.RGBA8,
            nvcv.TensorLayout.NHWC,
            [5, 16, 32, 4],
            np.uint8,
        ),
        (
            2,
            (38, 7),
            nvcv.Format.RGB8p,
            nvcv.TensorLayout.NCHW,
            [2, 3, 7, 38],
            np.uint8,
        ),
    ],
)
def test_tensor_creation_imagebatch_works(
    n, size, fmt, gold_layout, gold_shape, gold_dtype
):
    tensor = nvcv.Tensor(n, size, fmt)
    assert tensor.shape == gold_shape
    assert tensor.layout == gold_layout
    assert tensor.dtype == gold_dtype
    assert tensor.ndim == len(gold_shape)

    tensor = nvcv.Tensor(nimages=n, imgsize=size, format=fmt)
    assert tensor.shape == gold_shape
    assert tensor.layout == gold_layout
    assert tensor.dtype == gold_dtype
    assert tensor.ndim == len(gold_shape)


@t.mark.parametrize(
    "shape, dtype,layout",
    [
        ([5, 16, 32, 4], np.float32, nvcv.TensorLayout.NHWC),
        ([7, 3, 33, 11], np.complex64, nvcv.TensorLayout.NCHW),
        ([3, 11], np.int16, None),
        ([16, 32, 4], np.float32, nvcv.TensorLayout.HWC),
        ([32, 4], np.float32, nvcv.TensorLayout.WC),
        ([4, 32], np.float32, nvcv.TensorLayout.CW),
    ],
)
def test_tensor_creation_shape_works(shape, dtype, layout):
    tensor = nvcv.Tensor(shape, dtype, layout)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout == layout
    assert tensor.ndim == len(shape)

    tensor = nvcv.Tensor(layout=layout, shape=shape, dtype=dtype)
    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tensor.shape == shape
    assert tensor.ndim == len(shape)


@t.mark.parametrize(
    "shape,dtype",
    [
        ([3, 5, 7, 1], np.uint8),
        ([3, 5, 7, 1], np.int8),
        ([3, 5, 7, 1], np.uint16),
        ([3, 5, 7, 1], np.int16),
        ([3, 5, 7, 1], np.float32),
        ([3, 5, 7, 1], np.float64),
        ([3, 5, 7, 2], np.float32),
        ([3, 5, 7, 3], np.uint8),
        ([3, 5, 7, 4], np.uint8),
        ([3, 5, 7], np.csingle),
    ],
)
def test_wrap_numba_buffer(shape, dtype):
    tensor = nvcv.as_tensor(cuda.device_array(shape, dtype))
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout is None
    assert tensor.ndim == len(shape)


@t.mark.parametrize(
    "shape,dtype,layout",
    [
        ([3, 5, 7, 1], np.uint8, "NHWC"),
        ([3, 5, 7], np.uint8, "HWC"),
        ([3, 5, 7, 2], np.int16, "NHWC"),
        ([3, 5, 7, 2, 4, 2, 5], np.int16, "abcdefg"),
        ([3, 5], np.uint8, "HW"),
        # pybind11 has issues converting single characters to TensorLayout
        # ([5], np.uint8,"W"),
    ],
)
def test_wrap_numba_buffer_with_layout(shape, dtype, layout):
    tensor = nvcv.as_tensor(cuda.device_array(shape, dtype), layout)
    assert tensor.shape == shape
    assert tensor.dtype == dtype
    assert tensor.layout == layout
    assert tensor.ndim == len(shape)
