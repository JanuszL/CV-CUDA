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

# SDIR is the directory where this script is located
SDIR=$(dirname "$(readlink -f "$0")")

distro_name=$(lsb_release -is)
distro_ver=$(lsb_release -rs)

function version_le()
{
    [[ $(echo -e "$1\n$2" | sort -V | head -n1) = "$1" ]] && echo true
}

if ! which pre-commit || ! which shellcheck ; then
    echo 'pre-commit must be fully configured.'
    if [[ "$distro_name" = "Ubuntu" ]]; then
        if [[ $(version_le "$distro_ver" "18.04") ]]; then
            echo "Ubuntu v$distro_ver is too old, you need at least Ubuntu 20.04."
        elif [[ $(version_le "$distro_ver" "21.10") ]]; then
            echo "Try 'sudo apt-get install -y pip shellcheck && sudo pip install pre-commit'."
        else
            echo "Try 'sudo apt-get install -y pre-commit shellcheck'."
        fi
    else
        echo "Try installing pre-commit and shellcheck packaged from your distro"
    fi
    exit 1
fi

if ! which git-lfs ; then
    echo "git-lfs must be fully configured. Try 'apt-get install git-lfs'."
    exit 1
fi

cd "$SDIR"

# We use LFS
git lfs install

# allow-missing-config is useful when checking out an old commit or a branch that don't have pre-config configuration.
pre-commit install \
    --allow-missing-config \
    --install-hooks \
    -t pre-commit \
    -t pre-merge-commit \
    -t commit-msg
