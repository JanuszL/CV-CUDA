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

if [ $# = 0 ]; then
    # No arguments? Lint all code.
    echo "Linting all code in the repository =========================="
    pre-commit run -a
else
    from=$1
    if [ $# = 1 ]; then
        to=HEAD
    elif [ $# = 2 ]; then
        to=$2
    else
        echo "Invalid arguments"
        echo "Usage: $(basename "$0") [ref_from [ref_to]]"
        exit 1
    fi

    echo "Linting files touched from commit $from to $to =============="
    echo "Files to be linted:"
    git diff --stat $from..$to
    pre-commit run --from-ref $from --to-ref $to
fi
