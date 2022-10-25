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

extra_args=""
if [ -d $HOME/.vim ]; then
    extra_args="$extra_args -v $HOME/.vim:$HOME/.vim"
elif [ -f $HOME/.vimrc ]; then
    extra_args="$extra_args -v $HOME/.vimrc:$HOME/.vimrc"
fi

if [ -f $HOME/.gdbinit ]; then
    extra_args="$extra_args -v $HOME/.gdbinit:$HOME/.gdbinit"
fi

if [ -f /etc/sudoers ]; then
    extra_args="$extra_args -v /etc/sudoers:/etc/sudoers"
fi
if [ -d /etc/sudoers.d ]; then
    extra_args="$extra_args -v /etc/sudoers.d:/etc/sudoers.d"
fi

extra_cmds="true"

# Set up git user inside docker
git_user_name=$(git config --global user.name)
git_user_email=$(git config --global user.email)
if [[ "$git_user_name" && "$git_user_email" ]]; then
    extra_cmds="$extra_cmds && git config --global user.name '$git_user_name'"
    extra_cmds="$extra_cmds && git config --global user.email '$git_user_email'"
fi

# Run docker
# Notes:
#   - first and second cache mappings are for ccache and pre-commit respectively.
#   - pre-commit needs $HOME/.npm
docker run --gpus=all -ti \
    -v /etc/group:/etc/group:ro \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/shadow:/etc/shadow:ro \
    -v $HOME/.cache:/cache \
    -v $HOME/.cache:$HOME/.cache \
    -v $HOME/.npm:$HOME/.npm \
    -v /var/tmp:/var/tmp \
    -v $SDIR/..:$HOME/cvcuda \
    $extra_args \
    $IMAGE_URL_BASE/devel-linux:$TAG_IMAGE \
    /usr/bin/bash -c "mkdir -p $HOME && chown $USER:$USER $HOME && su - $USER -c \"$extra_cmds\" && su - $USER"
