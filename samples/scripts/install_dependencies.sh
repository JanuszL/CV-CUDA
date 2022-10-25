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

apt-get update && apt-get install -y --no-install-recommends build-essential \
    build-essential software-properties-common \
    && rm -rf /var/lib/apt/lists/*

add-apt-repository --yes ppa:ubuntu-toolchain-r/test
apt-get update && apt-get install -y --no-install-recommends gcc-11 g++-11 \
   && rm -rf /var/lib/apt/lists/*
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11
update-alternatives --set gcc /usr/bin/gcc-11
update-alternatives --set g++ /usr/bin/g++-11
pip3 install torch torchvision
