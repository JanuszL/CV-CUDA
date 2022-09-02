#!/bin/bash -e

# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Check if files that should be handled by LFS are being committed as
# LFS objects

lfs_files=$(echo "$@" | xargs git check-attr filter | grep 'filter: lfs$' | sed -e 's@: filter: lfs@@')

binary_files=''

for file in $lfs_files; do
    soft_sha=$(git hash-object -w $file)
    raw_sha=$(git hash-object -w --no-filters $file)

    if [ $soft_sha == $raw_sha ]; then
        binary_files="* $file\n$binary_files"
    fi
done

if [[ "$binary_files" ]]; then
    echo "The following files tracked by git-lfs are being committed as standard git objects:"
    echo -e "$binary_files"
    echo "Revert your changes and commit those with git-lfs installed."
    echo "In repo's root directory, run: sudo apt-get git-lfs && git lfs install"
    exit 1
fi
