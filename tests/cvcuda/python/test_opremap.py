# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cvcuda
import pytest as t


@t.mark.parametrize(
    "src_args, map_args",
    [
        (
            ((5, 16, 23, 4), nvcv.Type.U8, "NHWC"),
            ((5, 17, 13, 2), nvcv.Type.F32, "NHWC"),
        ),
        (
            ((13, 21, 1), nvcv.Type._3U8, "HWC"),
            ((23, 11, 1), nvcv.Type._2F32, "HWC"),
        ),
    ],
)
def test_op_remap(src_args, map_args):
    src = cvcuda.Tensor(*src_args)
    map = cvcuda.Tensor(*map_args)

    dst = cvcuda.remap(src, map)
    assert dst.layout == src.layout
    assert dst.dtype == src.dtype
    assert dst.shape[:-1] == map.shape[:-1]
    assert dst.shape[-1] == src.shape[-1]

    dst = cvcuda.Tensor(src.shape, src.dtype, src.layout)
    tmp = cvcuda.remap_into(dst, src, map)
    assert tmp is dst

    stream = cvcuda.Stream()
    dst = cvcuda.remap(
        src=src,
        map=map,
        src_interp=cvcuda.Interp.CUBIC,
        map_interp=cvcuda.Interp.LINEAR,
        map_type=cvcuda.Remap.RELATIVE_NORMALIZED,
        align_corners=True,
        border=cvcuda.Border.REFLECT101,
        border_value=1.0,
        stream=stream,
    )
    assert dst.layout == src.layout
    assert dst.dtype == src.dtype
    assert dst.shape == src.shape

    tmp = cvcuda.remap_into(
        dst=dst,
        src=src,
        map=map,
        src_interp=cvcuda.Interp.LINEAR,
        map_interp=cvcuda.Interp.CUBIC,
        map_type=cvcuda.Remap.ABSOLUTE_NORMALIZED,
        align_corners=False,
        border=cvcuda.Border.REPLICATE,
        border_value=[0.6, 0.7, 0.8, 0.9],
        stream=stream,
    )
    assert tmp is dst
