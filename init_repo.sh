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

if ! which pre-commit || ! which shellcheck ; then
	echo "pre-commit must be fully configured. Try 'apt-get install pre-commit shellcheck'."
	exit 1
fi

cd "$SDIR"

# allow-missing-config is useful when checking out an old commit or a branch that don't have pre-config configuration.
pre-commit install \
    --allow-missing-config \
    --install-hooks \
    -t pre-commit \
    -t pre-merge-commit \
    -t commit-msg \
    -t post-rewrite \
    -t post-checkout
