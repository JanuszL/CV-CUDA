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


def test_interpolation_type():
    assert nvcv.Interp.NEAREST != nvcv.Interp.LINEAR
    assert nvcv.Interp.LINEAR != nvcv.Interp.CUBIC
    assert nvcv.Interp.CUBIC != nvcv.Interp.AREA
    assert nvcv.Interp.AREA != nvcv.Interp.NEAREST
