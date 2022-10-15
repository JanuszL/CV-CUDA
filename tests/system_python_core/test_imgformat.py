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


@t.mark.parametrize(
    "format,gold_channels",
    [
        (nvcv.Format.RGBA8, 4),
        (nvcv.Format.RGB8, 3),
        (nvcv.Format._2S16, 2),
        (nvcv.Format.S8, 1),
        (nvcv.Format.NV12, 3),
    ],
)
def test_imgformat_numchannels(format, gold_channels):
    assert format.channels == gold_channels


@t.mark.parametrize(
    "format,gold_planes",
    [
        (nvcv.Format.RGBA8, 1),
        (nvcv.Format.RGB8, 1),
        (nvcv.Format._2S16, 1),
        (nvcv.Format.S8, 1),
        (nvcv.Format.NV12, 2),
    ],
)
def test_imgformat_planes(format, gold_planes):
    assert format.planes == gold_planes
