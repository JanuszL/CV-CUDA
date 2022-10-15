#!/bin/bash -e

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

if [ $# != 1 ]; then
    echo "Invalid arguments"
    echo "Usage: $(basename "$0") <C image format header path>"
    exit 1
fi

imgfmt_header=$1
shift

sed -n 's@^#define NVCV_IMAGE_FORMAT_\([0-9][^ ]\+\) NVCV_DETAIL.*@DEF_NUM(\1)@gp' $imgfmt_header
sed -n 's@^#define NVCV_IMAGE_FORMAT_\([^0-9][^ ]\+\) NVCV_DETAIL.*@DEF(\1)@gp' $imgfmt_header
