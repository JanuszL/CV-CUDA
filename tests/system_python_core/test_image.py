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

import pytest as t
import nvcv
import numpy as np
from numba import cuda


def test_image_creation_works():
    img = nvcv.Image((7, 5), nvcv.Format.NV12)
    assert img.width == 7
    assert img.height == 5
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.NV12


def test_image_creation_arg_keywords():
    img = nvcv.Image(size=(7, 5), format=nvcv.Format.NV12)
    assert img.width == 7
    assert img.height == 5
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.NV12


buffmt_common = [
    # packed formats
    ([5, 7, 1], np.uint8, nvcv.Format.U8),
    ([5, 7, 1], np.uint8, nvcv.Format.U8),
    ([5, 7, 1], np.uint8, nvcv.Format.U8),
    ([5, 7], np.uint8, nvcv.Format.U8),
    ([5, 7, 1], np.int8, nvcv.Format.S8),
    ([5, 7, 1], np.uint16, nvcv.Format.U16),
    ([5, 7, 1], np.int16, nvcv.Format.S16),
    ([5, 7, 2], np.int16, nvcv.Format._2S16),
    ([5, 7, 1], np.float32, nvcv.Format.F32),
    ([5, 7, 1], np.float64, nvcv.Format.F64),
    ([5, 7, 2], np.float32, nvcv.Format._2F32),
    ([5, 7, 3], np.uint8, nvcv.Format.RGB8),
    ([5, 7, 4], np.uint8, nvcv.Format.RGBA8),
    ([1, 5, 7], np.uint8, nvcv.Format.U8),
    ([1, 5, 7, 4], np.uint8, nvcv.Format.RGBA8),
    ([5, 7], np.csingle, nvcv.Format._2F32),
]


@t.mark.parametrize("shape,dt,format", buffmt_common)
def test_wrap_host_buffer_infer_imgformat(shape, dt, format):
    img = nvcv.Image(np.ndarray(shape, dt))
    assert img.width == 7
    assert img.height == 5
    assert img.format == format

    img = nvcv.as_image(cuda.device_array(shape, dt))
    assert img.width == 7
    assert img.height == 5
    assert img.format == format


@t.mark.parametrize(
    "shape,dt,format",
    buffmt_common
    + [
        ([5, 7, 1], np.uint8, nvcv.Format.Y8),
        ([5, 7, 3], np.uint8, nvcv.Format.BGR8),
        ([5, 7, 4], np.uint8, nvcv.Format.BGRA8),
    ],
)
def test_wrap_host_buffer_explicit_format(shape, dt, format):
    img = nvcv.Image(np.ndarray(shape, dt), format)
    assert img.width == 7
    assert img.height == 5
    assert img.format == format

    img = nvcv.as_image(cuda.device_array(shape, dt), format)
    assert img.width == 7
    assert img.height == 5
    assert img.format == format


buffmt2_common = [
    # packed formats
    (
        [([6, 8], np.uint8), ([3, 4, 2], np.uint8)],
        nvcv.Format.NV12_ER,
    )
]


@t.mark.parametrize("buffers,format", buffmt2_common)
@t.mark.xfail  # wrapping of multiple cuda buffers is failing
def test_wrap_host_buffer_infer_imgformat_multiple_planes(buffers, format):
    img = nvcv.Image([np.ndarray(buf[0], buf[1]) for buf in buffers])
    assert img.width == 8
    assert img.height == 6
    assert img.format == format

    img = nvcv.as_image([cuda.device_array(buf[0], buf[1]) for buf in buffers])
    assert img.width == 8
    assert img.height == 6
    assert img.format == format


@t.mark.parametrize("buffers,format", buffmt2_common)
@t.mark.xfail  # wrapping of multiple cuda buffers is failing
def test_wrap_host_buffer_explicit_format2(buffers, format):
    img = nvcv.Image([np.ndarray(buf[0], buf[1]) for buf in buffers], format)
    assert img.width == 8
    assert img.height == 6
    assert img.format == format

    img = nvcv.as_image([cuda.device_array(buf[0], buf[1]) for buf in buffers], format)
    assert img.width == 8
    assert img.height == 6
    assert img.format == format


@t.mark.parametrize(
    "shape,dt,planes,height,width,channels",
    [
        ([2, 7, 6], np.uint8, 2, 7, 6, 2),
        ([1, 2, 7, 6], np.uint8, 2, 7, 6, 2),
        ([2, 7, 3], np.uint8, 1, 2, 7, 3),
        ([1, 7, 3], np.uint8, 1, 1, 7, 3),
        ([7, 3], np.uint8, 1, 7, 3, 1),
        ([7, 1], np.uint8, 1, 7, 1, 1),
        ([1, 3], np.uint8, 1, 1, 3, 1),
        ([1, 1], np.uint8, 1, 1, 1, 1),
        ([5, 7, 3], np.uint8, 1, 5, 7, 3),
    ],
)
def test_wrap_host_buffer_infer_format_geometry(
    shape, dt, planes, height, width, channels
):
    img = nvcv.Image(np.ndarray(shape, dt))
    assert img.width == width
    assert img.height == height
    assert img.format.planes == planes
    assert img.format.channels == channels

    img = nvcv.as_image(cuda.device_array(shape, dt))
    assert img.width == width
    assert img.height == height
    assert img.format.planes == planes
    assert img.format.channels == channels


def test_wrap_host_buffer_arg_keywords():
    img = nvcv.Image(buffer=np.ndarray([5, 7], np.float32), format=nvcv.Format.F32)
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32

    img = nvcv.as_image(
        buffer=cuda.device_array([5, 7], np.float32), format=nvcv.Format.F32
    )
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32


def test_wrap_host_buffer_infer_format_arg_keywords():
    img = nvcv.Image(buffer=np.ndarray([5, 7], np.float32))
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32

    img = nvcv.as_image(buffer=cuda.device_array([5, 7], np.float32))
    assert img.size == (7, 5)
    assert img.format == nvcv.Format.F32


def test_wrap_host_image_with_format__buffer_has_unsupported_type():
    with t.raises(ValueError):
        nvcv.Image(np.array([1 + 2j, 4 + 7j]), nvcv.Format._2F32)


def test_wrap_host_image__buffer_has_unsupported_type():
    with t.raises(ValueError):
        nvcv.Image(np.array([1 + 2j, 4 + 7j]))


def test_wrap_host_image__format_and_buffer_type_mismatch():
    with t.raises(ValueError):
        nvcv.Image(np.array([1.4, 2.85]), nvcv.Format.U8)


def test_wrap_host_image__only_pitch_linear():
    with t.raises(ValueError):
        nvcv.Image(np.ndarray([6, 4], np.uint8), nvcv.Format.Y8_BL)


def test_wrap_host_image__css_with_one_plane_failure():
    with t.raises(ValueError):
        nvcv.Image(np.ndarray([6, 4], np.uint8), nvcv.Format.NV12)


@t.mark.parametrize(
    "shape",
    [
        (5, 3, 4),  # Buffer shape HCW not supported
        (5, 7, 4),  # Buffer shape doesn't correspond to image format
    ],
)
def test_wrap_host_image_with_format__invalid_shape(shape):
    with t.raises(ValueError):
        nvcv.Image(np.ndarray(shape, np.uint8), nvcv.Format.RGB8)


@t.mark.parametrize(
    "shape",
    [
        # When buffer's number of dimensions is 4, first dimension must be 1, not 2
        (
            2,
            3,
            4,
            5,
        ),
        # Number of dimensions must be between 1 and 4, not 5
        (1, 1, 15, 7, 1),
        # Number of dimensions must be between 1 and 4, not 0
        (0,),
        # Buffer shape not supported
        (8, 7, 9),
    ],
)
def test_wrap_host_invalid_dims(shape):
    with t.raises(ValueError):
        nvcv.Image(np.ndarray(shape))


@t.mark.parametrize(
    "s",
    [
        # Fastest changing dimension must be packed, i.e.,
        # have stride equal to 1 bytes(s), not 2
        (2 * 2, 2),
        # Buffer strides must all be >= 0
        (0, 1),
    ],
)
def test_wrap_host_invalid_strides(s):
    with t.raises(ValueError):
        nvcv.Image(
            np.ndarray(
                shape=(3, 2), strides=s, buffer=bytearray(s[0] * 3), dtype=np.uint8
            )
        )


@t.mark.parametrize(
    "shapes",
    [
        # When wrapping multiple buffers, buffers with 4
        # dimensions must have first dimension == 1, not 2
        [
            (3, 4),
            (2, 2, 3, 1),
        ],
        # Number of buffer#1's dimensions must be
        # between 1 and 4, not 5
        [
            (3, 4),
            (5, 2, 2, 3, 1),
        ],
    ],
)
def test_wrap_host_multiplane_invalid_dims(shapes):
    buffers = []
    for shape in shapes:
        buffers.append(np.ndarray(shape, np.uint8))

    with t.raises(ValueError):
        nvcv.Image(buffers)


def test_image_wrap_invalid_cuda_buffer():
    class NonCudaMemory(object):
        pass

    obj = NonCudaMemory()
    obj.__cuda_array_interface__ = {
        "shape": (1, 1),
        "typestr": "i",
        "data": (419, True),
        "version": 3,
    }

    with t.raises(RuntimeError):
        nvcv.as_image(obj)
