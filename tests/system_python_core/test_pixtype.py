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
    "type,dt",
    [
        (nvcv.Type.U8, np.uint8),
        (nvcv.Type.U8, np.dtype(np.uint8)),
        (nvcv.Type.S8, np.int8),
        (nvcv.Type.U16, np.uint16),
        (nvcv.Type.S16, np.int16),
        (nvcv.Type.U32, np.uint32),
        (nvcv.Type.S32, np.int32),
        (nvcv.Type.U64, np.uint64),
        (nvcv.Type.S64, np.int64),
        (nvcv.Type.F32, np.float32),
        (nvcv.Type.F64, np.float64),
        (nvcv.Type._2F32, np.complex64),
        (nvcv.Type._2F64, np.complex128),
        (nvcv.Type._3S8, np.dtype("3i1")),
        (nvcv.Type._4S32, np.dtype("4i")),
    ],
)
def test_pixtype_dtype(type, dt):
    assert type == dt

    t = nvcv.Type(dt)
    assert type == t
    assert dt == t
