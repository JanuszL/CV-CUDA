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


def test_border_type():
    assert nvcv.Border.CONSTANT != nvcv.Border.REPLICATE
    assert nvcv.Border.REPLICATE != nvcv.Border.REFLECT
    assert nvcv.Border.REFLECT != nvcv.Border.WRAP
    assert nvcv.Border.WRAP != nvcv.Border.REFLECT101
    assert nvcv.Border.REFLECT101 != nvcv.Border.CONSTANT
