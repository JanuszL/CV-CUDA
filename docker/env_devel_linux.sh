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

# shellcheck source=docker/config
. "$SDIR/config"

# Run docker
# Note: first and second cache mappings are for ccache and pre-commit respectively.
docker run --pull always --gpus=all -ti \
    -v $HOME/.cache:/cache \
    -v $HOME/.cache:$HOME/.cache \
    -v $SDIR/..:/cvcuda \
    $IMAGE_URL_BASE/devel-linux:$TAG_IMAGE
