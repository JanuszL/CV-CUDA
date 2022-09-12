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

if [ $# != 1 ]; then
    echo "Invalid arguments"
    echo "Usage: $(basename "$0") <container tag id>"
    exit 1
fi

tag_used=$1
shift

# shellcheck source=docker/config
. $SDIR/../docker/config

if [ "$TAG_IMAGE" != "$tag_used" ]; then
    echo "Tag of docker image used, $IMAGE_URL_BASE:$tag_used, must be $TAG_IMAGE. Please update the .gitlab-ci.yml" && false
fi
