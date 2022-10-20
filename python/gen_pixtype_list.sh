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
    echo "Usage: $(basename "$0") <C pixel type header path>"
    exit 1
fi

pixtype_header=$1
shift

sed -n 's@^#define NVCV_PIXEL_TYPE_\([0-9][^ ]\+\) \+NVCV_DETAIL.*@DEF_NUM(\1)@gp' $pixtype_header
sed -n 's@^#define NVCV_PIXEL_TYPE_\([^0-9][^ ]\+\) \+NVCV_DETAIL.*@DEF(\1)@gp' $pixtype_header
