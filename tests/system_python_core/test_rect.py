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


def test_recti_default():
    r = nvcv.RectI()
    assert r.x == 0
    assert r.y == 0
    assert r.width == 0
    assert r.height == 0


@t.mark.parametrize("x,y,w,h", [(0, 0, 0, 0), (10, -12, -45, 14)])
def test_recti_ctor(x, y, w, h):
    r = nvcv.RectI(x, y, w, h)
    assert r.x == x
    assert r.y == y
    assert r.width == w
    assert r.height == h

    r = nvcv.RectI(y=y, width=w, height=h, x=x)
    assert r.x == x
    assert r.y == y
    assert r.width == w
    assert r.height == h
