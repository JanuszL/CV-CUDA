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

# Builds samples
# Usage: build_samples.sh [build folder]

build_type="release"
build_dir="build"

if [[ $# -ge 1 ]]; then
   build_dir=$1
fi

 ./ci/build.sh $build_type $build_dir "-DBUILD_SAMPLES=ON -DBUILD_TESTS=OFF"
