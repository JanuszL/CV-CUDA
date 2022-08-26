#!/bin/bash -e

# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

args="$*"

do_push=0

if [[ $# == 1 && $1 == "--push" ]]; then
    do_push=1
    shift
elif [[ $# != 0 ]]; then
    echo "Usage: $(basename "$0") [--push]"
    exit 1
fi

cd "$SDIR"

# load up configuration variables
. ./config

cd build

image=$IMAGE_URL_BASE/build-linux:$TAG_IMAGE

docker build \
    --build-arg "VER_CUDA=$VER_CUDA" \
    --build-arg "VER_UBUNTU=$VER_UBUNTU" \
    . -t "$image"

if [[ $do_push == 1 ]]; then
    docker push "$image"
fi

cd ..

# must update all other dependent images
./update_devel_image.sh $args

cd ..
